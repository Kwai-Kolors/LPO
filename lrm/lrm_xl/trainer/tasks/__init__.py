
from hydra.core.config_store import ConfigStore

from trainer.tasks.step_sdxl_task import StepSDXLTaskConfig

cs = ConfigStore.instance()
cs.store(group="task", name="step_sdxl", node=StepSDXLTaskConfig)