# Size Annealing Hook for Deploy Resolution Fine-tuning
from mmengine.hooks import Hook
from mmdet.registry import HOOKS

@HOOKS.register_module()
class SizeAnnealingHook(Hook):
    """Anneal training resolution to the deploy resolution in the final epochs.

    Args:
        anneal_epoch (int): Epoch to start size annealing.
        deploy_scale (tuple[int, int]): Target (w, h) for deployment.
        deploy_ratio_range (tuple[float, float]): Ratio range at deploy size.
        once (bool): If True, switch only once; if False, switch every epoch >= anneal_epoch.
    """
    def __init__(self,
                 anneal_epoch=140,
                 deploy_scale=(640, 640),
                 deploy_ratio_range=(1.0, 1.0),
                 once=True):
        self.anneal_epoch = anneal_epoch
        self.deploy_scale = deploy_scale
        self.deploy_ratio_range = deploy_ratio_range
        self.once = once
        self._done = False

    def _unwrap_dataset(self, ds):
        """Unwrap common wrappers (RepeatDataset, ClassBalancedDataset, etc.)"""
        MAX_DEPTH = 5
        depth = 0
        while hasattr(ds, 'dataset') and depth < MAX_DEPTH:
            ds = ds.dataset
            depth += 1
        return ds

    def before_train_epoch(self, runner):
        """Switch to deploy resolution if we've reached anneal_epoch."""
        if runner.epoch < self.anneal_epoch:
            return
        if self.once and self._done:
            return

        ds = self._unwrap_dataset(runner.train_dataloader.dataset)
        # mmdet/mmengine pipelines are Compose objects
        pipeline = getattr(ds, 'pipeline', None)
        transforms = getattr(pipeline, 'transforms', None)

        if not transforms:
            runner.logger.warning('SizeAnnealingHook: no pipeline/transforms found.')
            return

        found = False
        for t in transforms:
            # RandomResize has attributes 'scale' and 'ratio_range'
            if t.__class__.__name__ == 'RandomResize' and hasattr(t, 'scale'):
                t.scale = self.deploy_scale
                if hasattr(t, 'ratio_range'):
                    t.ratio_range = self.deploy_ratio_range
                found = True
                break

        if found:
            runner.logger.info(
                f"🔧 Size Annealing → Using deploy size {self.deploy_scale}, "
                f"ratio_range={self.deploy_ratio_range} from epoch {runner.epoch}."
            )
            self._done = True
        else:
            runner.logger.warning('SizeAnnealingHook: RandomResize not found in train pipeline.')