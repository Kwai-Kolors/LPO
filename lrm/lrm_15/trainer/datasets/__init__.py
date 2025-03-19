from hydra.core.config_store import ConfigStore

from trainer.datasets.step_sd_hf_dataset import StepSDHFDatasetConfig

cs = ConfigStore.instance()
cs.store(group="dataset", name="step_sd", node=StepSDHFDatasetConfig)

