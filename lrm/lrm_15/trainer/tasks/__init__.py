
from hydra.core.config_store import ConfigStore

from trainer.tasks.step_sd_task import StepSDTaskConfig

cs = ConfigStore.instance()
cs.store(group="task", name="step_sd", node=StepSDTaskConfig)



