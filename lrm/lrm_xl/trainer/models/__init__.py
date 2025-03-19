from hydra.core.config_store import ConfigStore

from trainer.models.sdxl_base_preference_model import SDXLBasePreferenceModelConfig

cs = ConfigStore.instance()
cs.store(group="model", name="step_sdxl_base", node=SDXLBasePreferenceModelConfig)



