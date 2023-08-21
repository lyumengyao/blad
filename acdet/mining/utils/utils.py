import numpy as np
import torch
import time
import pickle
import mmcv
import torch
import torch.distributed as dist
from torch.autograd import grad


def sort_dict(valued_image_set, sort_key='value', reverse=False):
    sorted_valued_image_set = sorted(valued_image_set, key=lambda k: k[sort_key], reverse=reverse)
    return sorted_valued_image_set

def count_topk_image(sorted_valued_image_set, select_num, cnt_box=False):
    if cnt_box:
        boxcnts_cumsum = np.cumsum([d['cntbox'] for d in sorted_valued_image_set])
        return np.searchsorted(boxcnts_cumsum, select_num, side='right')
    else:
        return select_num


def select(valued_image_set, select_num, reverse, cnt_box=False):    
    select_num = int(select_num)  # box/image

    sorted_valued_image_set = sort_dict(valued_image_set, sort_key='value', reverse=reverse)
    select_img = count_topk_image(sorted_valued_image_set, select_num, cnt_box=cnt_box)
    selected_valued_image_set = sorted_valued_image_set[:select_img]
    remained_valued_image_set = sorted_valued_image_set[select_img:]
    return selected_valued_image_set, remained_valued_image_set


def stratified_sampling(preset_layers=10):
    def _stratified_sampling(valued_image_set, select_num, reverse, cnt_box=False):
        layers = preset_layers
        if len(valued_image_set) <= select_num:
            return valued_image_set, []
        
        select_num = int(select_num)
        select_num_per_layer = int(select_num / layers)
        img_per_layer = int(len(valued_image_set) / layers)
        selected_valued_image_set, remained_valued_image_set = [], []
        sorted_valued_image_set = sort_dict(valued_image_set, sort_key='value', reverse=reverse)

        for i in range(layers):
            if i == layers - 1:
                num_need_to_add = select_num - len(selected_valued_image_set)
                img_need_to_add = count_topk_image(sorted_valued_image_set[i * img_per_layer: ], num_need_to_add, cnt_box=cnt_box)
                remained_valued_image_set += sorted_valued_image_set[i * img_per_layer + img_need_to_add : ]
            else:
                img_need_to_add = count_topk_image(sorted_valued_image_set[i * img_per_layer: (i + 1)* img_per_layer], select_num_per_layer, cnt_box=cnt_box)
                remained_valued_image_set += sorted_valued_image_set[i * img_per_layer + img_need_to_add : (i + 1)* img_per_layer]
            selected_valued_image_set += sorted_valued_image_set[i * img_per_layer : i * img_per_layer + img_need_to_add]

        return selected_valued_image_set, remained_valued_image_set 
    
    return _stratified_sampling
