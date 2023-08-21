import argparse
import copy
import os
import os.path as osp
import time
import glob
import warnings

import mmcv
import torch
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed
from mmdet.models import build_detector
from mmdet.utils import collect_env

from acdet.datasets import build_dataset
from acdet.apis import get_root_logger, train_detector
from acdet.utils import AppendDictAction, patch_config, setup_sampler, setup_hooks
from acdet.mining import mining

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume', 
        action='store_true',
        help='whether to automatically resume from the latest ckpt in workdir')
    parser.add_argument(
        '--load-from-prev', 
        help='whether to resume from previous step latest checkpoint')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=AppendDictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options', 
        nargs='+', 
        action=AppendDictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--ratio',
        type=str, default='1/10',
        help='ratio of training data to use')
    parser.add_argument(
        '--initial-ratio',
        type=str, default='1/10',
        help='initial ratio of training data to random sample')
    parser.add_argument('--count-box', action='store_true', help='whether count #box instead of #images')
    parser.add_argument(
        '--model-result',
        help='result file/dir by previous trained model')
    parser.add_argument(
        '--score-thresh',
        type=float, default=0.0,
        help='threshold of score from model result')
    parser.add_argument(
        '--unlabel-count',
        action='store_true',
        help='get a balanced unlabeled set for MIAOD')
    parser.add_argument(
        '--mining-method',
        type=str, default='random',
        help='method to mine unlabeled data')
    parser.add_argument(
        '--training-setting',
        default='full',
        choices=['full', 'partial', 'mixed', 'semi', 'fullts'],
        help='partial for labeled+partial samples and mixed for labeled+partial+unlabeled')
    parser.add_argument('--balanced-sampler', action='store_true', help='whether balance sample_ratio w.r.t #samples')
    parser.add_argument('--epochs', type=int, default=0, help='train by #epochs')
    parser.add_argument(
        '--sorted-reverse',
        action='store_true',
        help='whether to reversely sort the active learning scores')
    parser.add_argument(
        '--stratified-sample',
        type=int,
        default=1,
        help='level number to use the stratified sampling')

    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()
    args.ratio = eval(args.ratio)
    args.initial_ratio = eval(args.initial_ratio)

    cfg = Config.fromfile(args.config)
    cfg.ratio = args.ratio
    cfg.initial_ratio = args.initial_ratio
    cfg.count_box = args.count_box
    cfg.score_thresh = args.score_thresh
    cfg.mining_method = args.mining_method
    if cfg.mining_method in ['learning_loss', 'miaod']:
        cfg.find_unused_parameters = True
    cfg.training_setting = args.training_setting
    cfg.balanced_sampler = args.balanced_sampler
    cfg.epochs = args.epochs
    top_select_methods = ['entropy', 'wbqbc', 'learning_loss', 'miaod', 'almdn', 'compas', 'boxcnt']
    if any(m in cfg.mining_method for m in top_select_methods):
        cfg.sorted_reverse = True
    if args.sorted_reverse:
        cfg.sorted_reverse = not cfg.sorted_reverse
    cfg.unlabel_count = args.unlabel_count
    cfg.stratified_sample = args.stratified_sample
    if args.model_result is not None:
        cfg.model_result = sorted(glob.glob(args.model_result))

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if cfg.training_setting != 'full' and cfg.mining_method != 'miaod':
        if cfg.ratio == 0:
            if cfg.training_setting == 'partial':
                cfg.training_setting = 'full'     # train as full dataset
                cfg.data.pop('sampler')
                cfg.custom_hooks.pop(2)
                cfg.evaluation.pop('type')
            elif cfg.training_setting == 'mixed':
                cfg.training_setting = 'semi'   # train as semi dataset
                cfg.data.train.type = "SemiDataset" # no partial labeled
                cfg.data.sampler.train.sample_ratio.pop(1)
        cfg = patch_config(cfg, wrapper=cfg.training_setting)

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    cfg.load_from_prev = args.load_from_prev
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = cfg.mining_method + '_' + osp.basename(args.config)

    if cfg.ratio == 0 and osp.exists(cfg.work_dir + "/labeled.json"):
        logger.info('********* Using saved splits for initialization *********')
        cfg.data.train.labeled.ann_file = cfg.work_dir + "/labeled.json"
        cfg.data.train.unlabeled.ann_file = cfg.work_dir + "/unlabeled.json"
        labeled_datasets = cfg.data.train.labeled
        unlabeled_datasets = cfg.data.train.unlabeled
    else:
        labeled_datasets, unlabeled_datasets = mining(logger, cfg)
    cfg, multiplier = setup_sampler(logger, cfg, labeled_datasets, unlabeled_datasets)
    cfg.optimizer.lr *= 1 / multiplier

    if args.load_from_prev:
        multiplier *= 0.3
        cfg.optimizer.lr /= 10
        cfg.lr_config.update(
            dict(
                warmup='linear',
                warmup_iters=500,
                warmup_ratio=0.001,
            ))
        cfg.lr_config.step = [cfg.lr_config.step[-1]]
    cfg.runner.max_iters = int(multiplier * cfg.runner.max_iters)
    cfg.lr_config.step = [int(multiplier * s) for s in cfg.lr_config.step]

    cfg = setup_hooks(cfg, multiplier)
    cfg.evaluation.interval = cfg.runner.max_iters//4
    cfg.checkpoint_config.interval = cfg.runner.max_iters//4
    logger.info(f'Train w/ {cfg.runner.max_iters} iters, scheduled at {cfg.lr_config.step}.')
    
    if cfg.training_setting == 'full':
        datasets = [build_dataset(labeled_datasets)]
    elif cfg.training_setting == 'fullts':
        cfg.data.train.labeled = labeled_datasets
        cfg.data.train.unlabeled = labeled_datasets
        datasets = [build_dataset(cfg.data.train)]
    else:
        cfg.data.train.labeled = labeled_datasets
        if cfg.mining_method == 'miaod':
            cfg.data.train.unlabeled.ann_file = cfg.work_dir + "/unlabeled_selection.json"
        else:
            cfg.data.train.unlabeled.ann_file = cfg.work_dir + "/unlabeled.json"
        if cfg.training_setting in ['mixed', 'partial']:
            cfg.data.train.mixed = unlabeled_datasets
            cfg.data.train.partial.ann_file = cfg.work_dir + "/partial.json"
        datasets = [build_dataset(cfg.data.train)]

    logger.info(f'Config:\n{cfg.pretty_text}')
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES)
        # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta) 


if __name__ == '__main__':
    main()
