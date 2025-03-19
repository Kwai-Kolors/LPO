from hydra.core.config_store import ConfigStore

from trainer.criterions.step_clip_criterion_xl import StepXLCLIPCriterionConfig


cs = ConfigStore.instance()
cs.store(group="criterion", name="step_clip_xl", node=StepXLCLIPCriterionConfig)