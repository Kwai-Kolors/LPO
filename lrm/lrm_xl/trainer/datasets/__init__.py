from hydra.core.config_store import ConfigStore

from trainer.datasets.step_sdxl_hf_dataset import StepSDXLHFDatasetConfig

cs = ConfigStore.instance()
cs.store(group="dataset", name="step_sdxl", node=StepSDXLHFDatasetConfig)