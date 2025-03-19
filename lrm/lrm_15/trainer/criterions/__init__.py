from hydra.core.config_store import ConfigStore

from trainer.criterions.step_clip_criterion import StepCLIPCriterionConfig


cs = ConfigStore.instance()
cs.store(group="criterion", name="step_clip", node=StepCLIPCriterionConfig)
