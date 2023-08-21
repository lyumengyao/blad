from copy import deepcopy
import shutil
import time
import argparse
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info

from .utils import *
from .methods import * 
from .datasets import *
from .registry import *


def mining(logger, cfg):
    save_dir = cfg.work_dir        
    
    full_datasets = cfg.data.train.full
    ann_file = full_datasets.ann_file
    labeled_datasets = cfg.data.train.labeled.copy()
    labeled_ann_file = labeled_datasets.ann_file
    labeled_datasets.ann_file = save_dir + "/labeled.json"
    if not cfg.mining_method.startswith('box_'):
        unlabeled_datasets = cfg.data.train.unlabeled.copy()
        unlabeled_ann_file = unlabeled_datasets.ann_file
        unlabeled_datasets.ann_file = save_dir + "/unlabeled.json"
    else:
        unlabeled_datasets = cfg.data.train.mixed.copy()
        unlabeled_ann_file = unlabeled_datasets.ann_file
        unlabeled_datasets.ann_file = save_dir + "/mixed.json"

    rank, _ = get_dist_info()
    if rank == 0:
        #if labeled_datasets.type == 'ActiveCocoDataset':
        if cfg.ratio == 0 or not cfg.mining_method.startswith('box_'):
            load_func = coco_load
            combine_func = coco_combine
            save_func = coco_save        
        else:        
            load_func = coco_load_obj
            save_func = coco_save_obj
        if cfg.get('resume_from') is None:
            if cfg.ratio == 0:
                logger.info(f'Randomly mine {cfg.initial_ratio} unlabeled data')
                labeled_image_set = []
                unlabeled_image_set, meta = coco_load(ann_file, config=cfg)
                cfg.sorted_reverse = False
                miner = MINERS.get('random')(logger, cfg)
                selected_valued_image_set, remained_valued_image_set = miner.run(unlabeled_image_set, cfg.initial_ratio)
            else:
                logger.info(f'Mine {cfg.ratio} unlabeled data by {cfg.mining_method}')
                labeled_image_set, meta = load_func(labeled_ann_file, config=cfg)
                unlabeled_image_set, _  = load_func(unlabeled_ann_file, config=cfg)
                miner = MINERS.get(cfg.mining_method)(logger, cfg)
                if cfg.mining_method.startswith('box_'):
                    selected_valued_image_set, remained_valued_image_set = deepcopy(labeled_image_set), deepcopy(unlabeled_image_set)
                    selected_valued_image_set, remained_valued_image_set = miner.run(selected_valued_image_set, \
                        remained_valued_image_set, cfg.ratio)
                else:
                    selected_valued_image_set, remained_valued_image_set = miner.run(unlabeled_image_set, cfg.ratio)
                    save_func(selected_valued_image_set, meta, save_dir + "/new_added_data.json")
                    selected_valued_image_set = combine_func([selected_valued_image_set, labeled_image_set])
            if cfg.ratio == 0 or not cfg.mining_method.startswith('box_'):
                logger.info(f'{len(selected_valued_image_set)} images are selected as labeled.')
                logger.info(f'{len(remained_valued_image_set)} images are selected as unlabeled.') 
                logger.info(f'{len(labeled_image_set)} images are used as labeled in previous step.')
                logger.info(f'{len(unlabeled_image_set)} images are used as unlabeled in previous step.')
            else:
                logger.info(f'{selected_valued_image_set.get_len()} images/boxes are selected as full labeled/unlabeled.')
                logger.info(f'{remained_valued_image_set.get_len()} images/boxes are selected as sparse labeled/unlabeled.') 
                logger.info(f'{labeled_image_set.get_len()} images/boxes are selected as labeled/unlabeled in previous step.')
                logger.info(f'{unlabeled_image_set.get_len()} images/boxes are selected as labeled/unlabeled in previous step.')
            save_func(selected_valued_image_set, meta, labeled_datasets.ann_file, labeled=True)
            unlabel_count = len(selected_valued_image_set) if cfg.unlabel_count else -1
            save_func(remained_valued_image_set, meta, unlabeled_datasets.ann_file, labeled=False, count=unlabel_count)
            if cfg.ratio == 0 and cfg.mining_method.startswith('box_'):
                shutil.copy(unlabeled_datasets.ann_file, save_dir + "/unlabeled.json")
            
    dist.barrier()
    return labeled_datasets, unlabeled_datasets