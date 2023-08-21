import glob
import os
import os.path as osp
import shutil
import types
from argparse import Action
import json

from mmcv.runner import BaseRunner, EpochBasedRunner, IterBasedRunner
from mmcv.utils import Config

from .signature import parse_method_info
from .vars import resolve


def find_latest_checkpoint(path, ext="pth"):
    if not osp.exists(path):
        return None
    if osp.exists(osp.join(path, f"latest.{ext}")):
        return osp.join(path, f"latest.{ext}")

    checkpoints = glob.glob(osp.join(path, f"*.{ext}"))
    if len(checkpoints) == 0:
        return None
    latest = -1
    latest_path = None
    for checkpoint in checkpoints:
        count = int(osp.basename(checkpoint).split("_")[-1].split(".")[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path

def find_prev_checkpoint(path, ext="pth"):
    step = int(path.split("/")[-1].split("_")[-1])
    if step == 0:
        return None
    prev_path = osp.join(*(path.split("/")[:-1]), f"step_{step - 1}")
    if not osp.exists(prev_path):
        raise NotImplementedError(f"Previous folder {prev_path} doesnt exist!")
    prev_latest_path = find_latest_checkpoint(prev_path, ext)
    if prev_latest_path is None:
        raise NotImplementedError(f"Previous checkpoint in {prev_path} doesnt exist!")
    
    return prev_latest_path

def patch_checkpoint(runner: BaseRunner):
    # patch save_checkpoint
    old_save_checkpoint = runner.save_checkpoint
    params = parse_method_info(old_save_checkpoint)
    default_tmpl = params["filename_tmpl"].default

    def save_checkpoint(self, out_dir, **kwargs):
        create_symlink = kwargs.get("create_symlink", True)
        filename_tmpl = kwargs.get("filename_tmpl", default_tmpl)
        # create_symlink
        kwargs.update(create_symlink=False)
        old_save_checkpoint(out_dir, **kwargs)
        if create_symlink:
            dst_file = osp.join(out_dir, "latest.pth")
            if isinstance(self, EpochBasedRunner):
                filename = filename_tmpl.format(self.epoch + 1)
            elif isinstance(self, IterBasedRunner):
                filename = filename_tmpl.format(self.iter + 1)
            else:
                raise NotImplementedError()
            filepath = osp.join(out_dir, filename)
            shutil.copy(filepath, dst_file)

    runner.save_checkpoint = types.MethodType(save_checkpoint, runner)
    return runner


def patch_runner(runner):
    runner = patch_checkpoint(runner)
    return runner


def setup_env(cfg):
    os.environ["WORK_DIR"] = cfg.work_dir


def patch_config(cfg, wrapper):
    if wrapper is not None or wrapper != 'full':
        cfg_dict = super(Config, cfg).__getattribute__("_cfg_dict").to_dict()
        cfg_dict["cfg_name"] = osp.splitext(osp.basename(cfg.filename))[0]
        cfg_dict = resolve(cfg_dict)
        cfg = Config(cfg_dict, filename=cfg.filename)
        # wrap for semi
        wrapper += "_wrapper"
        if cfg.get(wrapper, None) is not None:
            cfg.model = cfg.get(wrapper)
            cfg.pop(wrapper)
        # enable environment variables
    setup_env(cfg)
    return cfg


def setup_sampler(logger, cfg, labeled_datasets, unlabeled_datasets):
    world_size = len(cfg.gpu_ids)
    labeled_img_size = len(json.load(open(labeled_datasets.ann_file))['images'])
    partial_img_size = 0
    if cfg.training_setting in ['mixed', 'partial']:
        partial_img_size = len(json.load(open(osp.join(cfg.work_dir, "partial.json")))['images'])
    if cfg.epochs > 0:
        # max_iters = int(cfg.epochs * (labeled_img_size + partial_img_size) / 16 )
        max_iters = int(cfg.epochs * labeled_img_size / 16 )
        cfg.lr_config.step = [int(s / cfg.runner.max_iters * max_iters) for s in cfg.lr_config.step]
        cfg.runner.max_iters = max_iters
    if cfg.data.get('sampler') is not None:
        # if cfg.balanced_sampler:
        unlabel_img_size = cfg.data.valid_len - labeled_img_size - partial_img_size
        sample_ratio = [labeled_img_size]
        if cfg.training_setting == "partial":
            sample_ratio.extend([partial_img_size])
            wns = ['partial_weight']
        elif cfg.training_setting == "semi":
            sample_ratio.extend([unlabel_img_size])
            wns = ['unsup_weight']
        elif cfg.training_setting == "mixed":
            sample_ratio.extend([partial_img_size, unlabel_img_size])
            wns = ['partial_weight', 'unsup_weight']
        if not cfg.balanced_sampler:
            sample_ratio = cfg.data.sampler.train.sample_ratio
        else:
            min_sr = min(sample_ratio)
            sample_ratio = [int(sr / min_sr) if sr / min_sr <= 5 else 5 for sr in sample_ratio]
        logger.info(f'Sample ratio: {sample_ratio}')
        if True: # depends on sample ratio # cfg.balanced_sampler:
            for r, wn in zip(sample_ratio[1:], wns):
                cfg.model.train_cfg[wn] = r/sample_ratio[0]
        cfg.data.sampler.train.sample_ratio = sample_ratio
        if cfg.mining_method == 'miaod':
            samples_per_gpu = 2 * cfg.data.samples_per_gpu
            multiplier = 16 / cfg.data.samples_per_gpu / world_size
            cfg.data.samples_per_gpu = samples_per_gpu # 1:1
        else:
            samples_per_gpu = 2 * sum(sample_ratio)
            cfg.data.samples_per_gpu = samples_per_gpu
            multiplier = 16 / (2 * sample_ratio[0]) / world_size
        cfg.data.workers_per_gpu = min(samples_per_gpu, 8)
    else:
        cfg.data.samples_per_gpu = 2
        multiplier = 16 / cfg.data.samples_per_gpu / world_size
    return cfg, multiplier


def setup_hooks(cfg, multiplier):
    # method-specific handling
    if cfg.mining_method == 'learning_loss':
        assert cfg.custom_hooks[1].type == 'DropLearningLossIterBasedHook'
        cfg.custom_hooks[1].freeze_iter = int(multiplier * cfg.custom_hooks[1].freeze_iter)
    if cfg.mining_method == 'miaod':
        assert cfg.training_setting == "semi"
        # assert args.load_from_prev is None
        stage_epochs = [d['epochs'] for d in cfg.custom_hooks[-1].stages]
        last_stage_epochs = stage_epochs[-1]
        stage_epochs = sum(stage_epochs)
        if False:
            # same as author's code, iters depend on size of labeled samples
            current_size = len(json.load(open(labeled_datasets.ann_file))['images'])
            cfg.runner.max_iters = int(cfg.repeat_times*stage_epochs*current_size/world_size/cfg.data.samples_per_gpu)
        cfg.lr_config.step = int(cfg.runner.max_iters * (1-last_stage_epochs/stage_epochs*(1/3)))
    return cfg

class AppendDictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options should
    be passed as comma separated values, i.e KEY=V1,V2,V3
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        return val

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None)
        if items is None:
            items = {}
        # items = _copy_items(items)
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            val = [self._parse_int_float_bool(v) for v in val.split(',')]
            if len(val) == 1:
                val = val[0]
            items[key] = val
        setattr(namespace, self.dest, items)

