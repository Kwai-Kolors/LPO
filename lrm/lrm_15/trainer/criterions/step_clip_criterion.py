from dataclasses import dataclass
import torch
from omegaconf import II
from torch.nn.modules.loss import _Loss


@dataclass
class StepCLIPCriterionConfig:
    _target_: str = "trainer.criterions.step_clip_criterion.StepCLIPCriterion"
    is_distributed: bool = True
    label_0_column_name: str = II("dataset.label_0_column_name")
    label_1_column_name: str = II("dataset.label_1_column_name")

    input_ids_column_name: str = II("dataset.input_ids_column_name")
    pixels_0_column_name: str = II("dataset.pixels_0_column_name")
    pixels_1_column_name: str = II("dataset.pixels_1_column_name")
    num_examples_per_prompt_column_name: str = II("dataset.num_examples_per_prompt_column_name")
    timestep_column_name: str = II("dataset.timestep_column_name")
    # in_batch_negatives: bool = False
    # both_loss: bool = False
    loss_type: str = "pair"  # batch, pair, or both
    batch_coeff: float = 1.0
    aux_loss_coeff: float = 1.0
    pass


class StepCLIPCriterion(_Loss):
    def __init__(self, cfg: StepCLIPCriterionConfig):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def get_features(model, input_ids, pixels_0_values, pixels_1_values, timesteps):
        all_pixel_values = torch.cat([pixels_0_values, pixels_1_values], dim=0)
        timesteps = timesteps.reshape(-1, 2)
        timesteps = torch.cat([timesteps[:,0], timesteps[:, 1]])
        # timesteps = torch.cat([timesteps, timesteps], dim=0)
        text_features, all_image_features = model(text_inputs=input_ids, image_inputs=all_pixel_values, time_cond=timesteps)
        all_image_features = all_image_features / all_image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_0_features, image_1_features = all_image_features.chunk(2, dim=0)
        return image_0_features, image_1_features, text_features

    @staticmethod
    def gather_features(features):
        all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
        return all_features

    def calc_loss(
            self,
            text_features,
            image_0_features,
            image_1_features,
            logit_scale,
            label_0,
            label_1,
            num_examples_per_prompt,
            timesteps,
            *args,
            **kwargs
    ):
        device = image_0_features.device

        # gather features
        if self.cfg.is_distributed:
            image_0_features = self.gather_features(image_0_features)
            image_1_features = self.gather_features(image_1_features)
            text_features = self.gather_features(text_features)
            label_0 = self.gather_features(label_0)
            label_1 = self.gather_features(label_1)
            num_examples_per_prompt = self.gather_features(num_examples_per_prompt)
            timesteps = self.gather_features(timesteps)

        # calc logits 
        all_image_features = torch.cat([image_0_features, image_1_features], dim=0)  # (2 * batch_size, dim)
        logits_per_image = logit_scale * all_image_features @ text_features.T
        image_0_logits, image_1_logits = logits_per_image.chunk(2, dim=0)  # ni * np
        text_logits = logit_scale * text_features @ all_image_features.T  # np * 2ni

        if self.cfg.loss_type == "batch":
            # get labels
            num_images = all_image_features.shape[0]
            image_labels = torch.arange(num_images, device=device, dtype=torch.long)
            image_0_labels, image_1_labels = image_labels.chunk(2, dim=0)
            num_texts = text_features.shape[0]
            text_labels = torch.arange(num_texts, device=device, dtype=torch.long)

            # image loss - we want to increase the logits of the preferred image to the text
            image_0_loss = torch.nn.functional.cross_entropy(image_0_logits, text_labels, reduction="none")
            image_1_loss = torch.nn.functional.cross_entropy(image_1_logits, text_labels, reduction="none")
            # if we have a tie, we will increase both images equally, and average so the image loss of each example is
            # proportional
            # image-text contrastive learning
            batch_image_loss = label_0 * image_0_loss + label_1 * image_1_loss

            # text loss - we want to increase the logits of the text to the preferred image
            text_0_loss = torch.nn.functional.cross_entropy(text_logits, image_0_labels, reduction="none")
            text_1_loss = torch.nn.functional.cross_entropy(text_logits, image_1_labels, reduction="none")
            
            # if we have a tie we want the logits of for each image to be equal
            batch_text_loss = label_0 * text_0_loss + label_1 * text_1_loss
            # we want the ideal loss to be 0, currently, if there is a tie, it is 0.5 * log(0.5) + 0.5 * log(0.5)
            # so we add log(0.5) to the loss
            is_tie = (label_0 == label_1).float()
            is_tie *= torch.log(torch.tensor(0.5, device=device))
            batch_text_loss += is_tie
            
            loss = (batch_image_loss + batch_text_loss) / 2

        elif self.cfg.loss_type == "pair":
            text_0_logits, text_1_logits = text_logits.chunk(2, dim=-1)
            index = torch.arange(text_0_logits.shape[0], device=device, dtype=torch.long)
            text_0_logits = text_0_logits[index, index]
            text_1_logits = text_1_logits[index, index]
            text_logits = torch.stack([text_0_logits, text_1_logits], dim=-1)
            text_0_labels = torch.zeros(text_logits.shape[0], device=device, dtype=torch.long)
            text_1_labels = text_0_labels + 1
            text_0_loss = torch.nn.functional.cross_entropy(text_logits, text_0_labels, reduction="none")
            text_1_loss = torch.nn.functional.cross_entropy(text_logits, text_1_labels, reduction="none")
            
            # if we have a tie we want the logits of for each image to be equal
            text_loss = label_0 * text_0_loss + label_1 * text_1_loss
            # we want the ideal loss to be 0, currently, if there is a tie, it is 0.5 * log(0.5) + 0.5 * log(0.5)
            # so we add log(0.5) to the loss
            is_tie = (label_0 == label_1).float()
            is_tie *= torch.log(torch.tensor(0.5, device=device))
            text_loss += is_tie
            
            loss = text_loss
            
        elif self.cfg.loss_type == "both":
            # get labels
            num_images = all_image_features.shape[0]
            image_labels = torch.arange(num_images, device=device, dtype=torch.long)
            image_0_labels, image_1_labels = image_labels.chunk(2, dim=0)
            num_texts = text_features.shape[0]
            text_labels = torch.arange(num_texts, device=device, dtype=torch.long)

            # image loss - we want to increase the logits of the preferred image to the text
            image_0_loss = torch.nn.functional.cross_entropy(image_0_logits, text_labels, reduction="none")
            image_1_loss = torch.nn.functional.cross_entropy(image_1_logits, text_labels, reduction="none")
            # if we have a tie, we will increase both images equally, and average so the image loss of each example is
            # proportional
            # image-text contrastive learning
            batch_image_loss = label_0 * image_0_loss + label_1 * image_1_loss

            # text loss - we want to increase the logits of the text to the preferred image
            text_0_loss = torch.nn.functional.cross_entropy(text_logits, image_0_labels, reduction="none")
            text_1_loss = torch.nn.functional.cross_entropy(text_logits, image_1_labels, reduction="none")
            
            # if we have a tie we want the logits of for each image to be equal
            batch_text_loss = label_0 * text_0_loss + label_1 * text_1_loss
            # we want the ideal loss to be 0, currently, if there is a tie, it is 0.5 * log(0.5) + 0.5 * log(0.5)
            # so we add log(0.5) to the loss
            is_tie = (label_0 == label_1).float()
            is_tie *= torch.log(torch.tensor(0.5, device=device))
            batch_text_loss += is_tie
            
            batch_loss = (batch_image_loss + batch_text_loss) / 2
            
            text_0_logits, text_1_logits = text_logits.chunk(2, dim=-1)
            index = torch.arange(text_0_logits.shape[0], device=device, dtype=torch.long)
            text_0_logits = text_0_logits[index, index]
            text_1_logits = text_1_logits[index, index]
            text_logits = torch.stack([text_0_logits, text_1_logits], dim=-1)
            text_0_labels = torch.zeros(text_logits.shape[0], device=device, dtype=torch.long)
            text_1_labels = text_0_labels + 1
            text_0_loss = torch.nn.functional.cross_entropy(text_logits, text_0_labels, reduction="none")
            text_1_loss = torch.nn.functional.cross_entropy(text_logits, text_1_labels, reduction="none")
            
            # if we have a tie we want the logits of for each image to be equal
            text_loss = label_0 * text_0_loss + label_1 * text_1_loss
            # we want the ideal loss to be 0, currently, if there is a tie, it is 0.5 * log(0.5) + 0.5 * log(0.5)
            # so we add log(0.5) to the loss
            is_tie = (label_0 == label_1).float()
            is_tie *= torch.log(torch.tensor(0.5, device=device))
            text_loss += is_tie
            
            loss = text_loss + self.cfg.batch_coeff * batch_loss

        # some prompts have lots of interactions, we want weight them accordingly
        absolute_example_weight = 1 / num_examples_per_prompt
        denominator = absolute_example_weight.sum()
        weight_per_example = absolute_example_weight / denominator
        loss *= weight_per_example

        # done  weight loss for timestep comparison by using different timesteps as identifiers
        timesteps = timesteps.reshape(-1, 2)
        flag = timesteps[:, 0] != timesteps[:, 1]
        aux_weight = torch.tensor([1]*loss.shape[0], device=loss.device, dtype=loss.dtype)
        aux_weight[flag] = self.cfg.aux_loss_coeff

        loss *= aux_weight

        loss = loss.sum()
        return loss

    def forward(self, model, batch):
        image_0_features, image_1_features, text_features = self.get_features(
            model,
            batch[self.cfg.input_ids_column_name],
            batch[self.cfg.pixels_0_column_name],
            batch[self.cfg.pixels_1_column_name],
            batch[self.cfg.timestep_column_name],
        )
        loss = self.calc_loss(
            text_features,
            image_0_features,
            image_1_features,
            model.logit_scale.exp(),
            batch[self.cfg.label_0_column_name],
            batch[self.cfg.label_1_column_name],
            batch[self.cfg.num_examples_per_prompt_column_name],
            batch[self.cfg.timestep_column_name],
        )
        return loss
