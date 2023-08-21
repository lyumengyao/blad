from .models import *
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from torch.nn import Dropout

class MCDropout(Dropout):
    def __init__(self, **kwargs):
        kwargs.pop('in_channels')
        super().__init__(**kwargs)

PLUGIN_LAYERS.register_module(name="MCDropout", module=MCDropout)
