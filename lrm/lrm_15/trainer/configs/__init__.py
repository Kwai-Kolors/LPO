from hydra.core.config_store import ConfigStore

from trainer.configs.configs import TrainerConfig
from trainer.configs.step_sd_configs import StepSDTrainerConfig

cs = ConfigStore.instance()
cs.store(name="base_config", node=TrainerConfig)
cs.store(name="step_sd_config", node=StepSDTrainerConfig)