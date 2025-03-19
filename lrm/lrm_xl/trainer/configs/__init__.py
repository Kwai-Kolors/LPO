from hydra.core.config_store import ConfigStore

from trainer.configs.configs import TrainerConfig
from trainer.configs.step_sdxl_configs import StepSDXLTrainerConfig

cs = ConfigStore.instance()
cs.store(name="base_config", node=TrainerConfig)
cs.store(name="step_sdxl_base_config", node=StepSDXLTrainerConfig)