import torch
from .builder import COMPARE_FUNCS


sigma_tensor = torch.tensor([0.0206, 0.0406, 0.1673, 0.2144, 0.2509, 0.2836, 0.3144, 0.3442, 0.3732,
    0.4016, 0.4293, 0.4565, 0.4829, 0.5087, 0.5336, 0.5576, 0.5807, 0.6029,
    0.6241, 0.6444])
min_sigma = sigma_tensor[0]
max_sigma = sigma_tensor[-1]

variance_tensor = torch.tensor([0.0004, 0.0016, 0.0280, 0.0460, 0.0630, 0.0804, 0.0988, 0.1185, 0.1393,
        0.1613, 0.1843, 0.2084, 0.2332, 0.2588, 0.2847, 0.3109, 0.3372, 0.3635,
        0.3895, 0.4153])

min_variance = variance_tensor[0]
max_variance = variance_tensor[-1]

min_timesteps = 1
max_timesteps = 951

def dynamic_threshold_sigma(threshold_min, threshold_max, timesteps):
    thresh_tensor = (sigma_tensor - min_sigma) / (max_sigma - min_sigma) * (threshold_max - threshold_min) + threshold_min
    thresh_tensor = thresh_tensor.to(timesteps.device)
    timestep_index = timesteps // 50
    threshold = torch.gather(thresh_tensor, 0, timestep_index)
    return threshold


def dynamic_threshold_variance(threshold_min, threshold_max, timesteps):
    thresh_tensor = (variance_tensor - min_variance) / (max_variance - min_variance) * (threshold_max - threshold_min) + threshold_min
    thresh_tensor = thresh_tensor.to(timesteps.device)
    timestep_index = timesteps // 50
    threshold = torch.gather(thresh_tensor, 0, timestep_index)
    return threshold


def dynamic_threshold_linear(threshold_min, threshold_max, timesteps):
    threshold = (timesteps - min_timesteps) / (max_timesteps - min_timesteps) * (threshold_max - threshold_min) + threshold_min
    return threshold



@COMPARE_FUNCS.register_module()
def preference_score_compare(scores, threshold, timesteps=None, dynamic_threshold=None, threshold_min=0.2, threshold_max=0.5):
    # scores: num_sample_per_step, b
    scores, indices = torch.sort(scores, dim=0, descending=True)
    # 2, b
    indices = indices[[0, -1], :]
    scores = scores[[0, -1], :]
    scores = scores.softmax(dim=0)
    # b
    if dynamic_threshold is not None:
        assert timesteps is not None, "Dynamic threshold needs timesteps"
        if dynamic_threshold == "linear":
            threshold = dynamic_threshold_linear(threshold_min, threshold_max, timesteps)
        elif dynamic_threshold == "sigma":
            threshold = dynamic_threshold_sigma(threshold_min, threshold_max, timesteps)
        elif dynamic_threshold == "variance":
            threshold = dynamic_threshold_variance(threshold_min, threshold_max, timesteps)
        else:
            raise NotImplementedError(f"the {dynamic_threshold} is not supported")
    valid_samples = scores[0] - scores[1] > threshold
    return indices, valid_samples


if __name__ == "__main__":
    timesteps = torch.arange(0, 1000, 50, device='cuda')
    print(timesteps)
    threshold1 = dynamic_threshold_sigma(0.35, 0.5, timesteps)
    threshold2 = dynamic_threshold_linear(0.35, 0.5, timesteps)
    threshold3 = dynamic_threshold_variance(0.35, 0.5, timesteps)
    print(timesteps)
    print(threshold1)
    print(threshold2)
    print(threshold3)
