import abc
import json
import numpy as np
import io
from ..utils import *
from ..registry import MINERS


class MiningAlgorithm(object):
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        cntbox = config.get('count_box', False)
        if self.config.ratio != 0:
            self.select_func = lambda x, y, z: stratified_sampling(self.config.stratified_sample)(x, y, z, cnt_box=cntbox)
        else:
            self.select_func = lambda x, y, z: select(x, y, z, cnt_box=cntbox)

    def run(self, unlabeled_data, ratio):
        score_thresh = self.config.get("score_thresh", 0)
        score_dict = self.load(score_thresh)
        selected_data, remained_data = self.mining(unlabeled_data, score_dict, ratio)
        return selected_data, remained_data

    @abc.abstractmethod
    def load(self, score_thresh=0):
        pass

    @abc.abstractmethod
    def mining(self, unlabeled_data, score_dict, ratio):
        pass


@MINERS.register_module(name='random')
class RandomMiningAlgorithm(MiningAlgorithm):
    def load(self, score_thresh=0):
        return None

    def mining(self, unlabeled_data, score_dict, ratio):
        for data in unlabeled_data:
            data['value'] = np.random.rand()
        
        # if the ratio is large than 1 then use the ratio as select_num
        select_num = ratio
        if ratio <= 1:
            select_num = int(ratio * 1.0 * len(unlabeled_data))

        return self.select_func(unlabeled_data, select_num, self.config.sorted_reverse)
