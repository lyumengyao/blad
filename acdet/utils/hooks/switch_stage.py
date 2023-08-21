import numpy as np

from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook

@HOOKS.register_module()
class SwitchStageIterBasedHook(Hook):
    def __init__(self, stages):
        self.stages = stages
        self.switch_points = np.cumsum([0] + [stage['epochs'] for stage in stages])
        self.switch_points = self.switch_points / self.switch_points[-1]

    def before_train_iter(self, runner):
        # Freeze the learning loss module
        switch_iters = getattr(self, 'switch_iters', False)
        if not switch_iters:
            self.switch_iters = [int(i) for i in runner.max_iters * self.switch_points]
        if runner.iter in self.switch_iters[:-1]:
            stage = self.stages[self.switch_iters.index(runner.iter)]

            model = runner.model
            if is_module_wrapper(model):
                model = model.module
            setattr(model, 'loss_type', stage['loss_type'])

            if stage.get('params_learn') is not None:
                for name, value in model.named_parameters():
                    value.requires_grad = True if name in stage['params_learn'] else False
            elif stage.get('params_freeze') is not None:
                for name, value in model.named_parameters():
                    value.requires_grad = False if name in stage['params_freeze'] else True
            else:
                for name, value in model.named_parameters():
                    value.requires_grad = True