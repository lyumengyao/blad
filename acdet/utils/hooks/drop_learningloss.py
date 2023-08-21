from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook

@HOOKS.register_module()
class DropLearningLossIterBasedHook(Hook):
    def __init__(self, freeze_iter):
        self.freeze_iter = freeze_iter

    def before_train_iter(self, runner):
        # Freeze the learning loss module
        if runner.iter == self.freeze_iter:
            model = runner.model
            if is_module_wrapper(model):
                model = model.module
            setattr(model.roi_head.bbox_head, "drop_ll", True)