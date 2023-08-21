import json
import random
from statistics import mean, median, variance, pstdev

from ..registry import MINERS
from .mining_algorithm import MiningAlgorithm


@MINERS.register_module(name='almdn')
class ALMDNMiningAlgorithm(MiningAlgorithm):
    def load(self, score_thresh):
        image_dict = {}
        if len(self.config.model_result) == 1:
            self.config.model_result = self.config.model_result[0]
        data = json.load(open(self.config.model_result, 'r'))
        for annotation in data:
            if annotation['score'] < score_thresh:
                continue
            image_id = annotation['image_id']
            if image_id not in image_dict:
                image_dict[image_id] = dict(
                    list_conf_al=[],
                    list_conf_ep=[],
                    list_loc_al=[],
                    list_loc_ep=[],
                )
            instance_dict = image_dict[image_id]
            instance_dict['list_conf_al'].append(annotation['cls_al_un'])
            instance_dict['list_conf_ep'].append(annotation['cls_ep_un'])
            instance_dict['list_loc_al'].append(max(annotation['reg_al_un']))
            instance_dict['list_loc_ep'].append(max(annotation['reg_ep_un']))
        def get_stats(list_of_dict, key):
            values = [i for img_dict in list_of_dict for i in img_dict[key]]
            return mean(values), pstdev(values)
        self.conf_al_mean, self.conf_al_std = get_stats(image_dict.values(), 'list_conf_al')
        self.conf_ep_mean, self.conf_ep_std = get_stats(image_dict.values(), 'list_conf_ep')
        self.loc_al_mean, self.loc_al_std = get_stats(image_dict.values(), 'list_loc_al')
        self.loc_ep_mean, self.loc_ep_std = get_stats(image_dict.values(), 'list_loc_ep')
        return image_dict

    def value(self, data_dict):
        conf_al = (max(data_dict['list_conf_al']) - self.conf_al_mean) / self.conf_al_std
        conf_ep = (max(data_dict['list_conf_ep']) - self.conf_ep_mean) / self.conf_ep_std
        loc_al = (max(data_dict['list_loc_al']) - self.loc_al_mean) / self.loc_al_std
        loc_ep = (max(data_dict['list_loc_ep']) - self.loc_ep_mean) / self.loc_ep_std
        return max([conf_al, conf_ep, loc_al, loc_ep])

    def mining(self, unlabled_data, score_dict, ratio):
        scored_data, unscored_data = [], []
        for data in unlabled_data:
            image_id = int(data['info']['id'])
            if image_id not in score_dict:
                unscored_data.append(data)
                continue
            data['value'] = self.value(score_dict[image_id])
            scored_data.append(data)
        self.logger.info(f'{len(scored_data)} images\' score is higher than score_thresh')
        self.logger.info(f'{len(unscored_data)} images\' score is lower than score_thresh')
        if ratio <= 1:
            to_select_num = int(ratio * len(unlabled_data))
        else:
            to_select_num = int(ratio)
        if len(scored_data) >= to_select_num:
            assert self.config.sorted_reverse
            selected_data, remained_data = self.select_func(scored_data, to_select_num, self.config.sorted_reverse)
            remained_data.extend(unscored_data)
        else:
            random.shuffle(unscored_data)
            to_add_num = to_select_num - len(scored_data)
            selected_data = scored_data + unscored_data[:to_add_num]
            remained_data = unscored_data[to_add_num:]
        return selected_data, remained_data