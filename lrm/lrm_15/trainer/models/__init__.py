from hydra.core.config_store import ConfigStore

from trainer.models.sd15_preference_model import SD15PreferenceModelConfig

cs = ConfigStore.instance()
cs.store(group="model", name="step_sd15", node=SD15PreferenceModelConfig)