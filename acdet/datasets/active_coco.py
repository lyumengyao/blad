import os.path as osp
import tempfile
import numpy as np

import mmcv
from mmdet.datasets import DATASETS, CocoDataset


@DATASETS.register_module()
class ActiveCocoDataset(CocoDataset):

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    # load gts for debug
    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_scores = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])
                if 'score' not in ann.keys():
                    gt_scores.append(1)
                else: 
                    gt_scores.append(float(ann['score']))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            scores=gt_scores)

        return ann

    def _coreset2json(self, results):
        """Convert detection coreset value to COCO json style."""
        json_results = []
        for rank, idx in enumerate(results):
            img_id = self.img_ids[idx]
            data = dict()
            data['image_id'] = img_id
            data['coreset'] = rank
            json_results.append(data)
        return json_results      

    def _ll2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            data = dict()
            data['image_id'] = img_id
            data['learning_loss'] = result
            json_results.append(data)
        return json_results

    def _miaod2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            data = dict()
            data['image_id'] = img_id
            data['miaod'] = result
            json_results.append(data)
        return json_results

    def _wbqbc2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            data = dict()
            data['image_id'] = img_id
            data['wbqbc'] = result
            json_results.append(data)
        return json_results

    def _almdn2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    res = bboxes[i]
                    data['bbox'] = self.xyxy2xywh(res)
                    data['score'] = float(res[4])
                    data['cls_al_un'] = float(res[5])
                    data['cls_ep_un'] = float(res[6])
                    data['reg_al_un'] = res[7:11].tolist()
                    data['reg_ep_un'] = res[11:].tolist()
                    json_results.append(data)
        return json_results

    def _compas2json(self, results):
        """Convert augmented detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            data_list = results[idx]
            for data in data_list:
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(data['bbox'])
                # data['category_id'] = self.cat_ids[data['category_id']]
                json_results.append(data)
        return json_results
        
    def results2json(self, results, outfile_prefix, **kwargs):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()

        if "coreset" in kwargs:
            json_results = self._coreset2json(results)
            result_files['coreset'] = f'{outfile_prefix}.coreset.json'
            mmcv.dump(json_results, result_files['coreset'])
        elif "learning_loss" in kwargs:
            json_results = self._ll2json(results)
            result_files['learning_loss'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['learning_loss'])
        elif "miaod" in kwargs:
            json_results = self._miaod2json(results)
            result_files['miaod'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['miaod'])
        elif "wbqbc" in kwargs:
            json_results = self._wbqbc2json(results)
            result_files['wbqbc'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['wbqbc'])
        elif "almdn" in kwargs:
            json_results = self._almdn2json(results)
            result_files['almdn'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['almdn'])
        elif "box_compas" in kwargs:
            json_results = self._compas2json(results)
            result_files['compas'] = f'{outfile_prefix}.compas.json'
            mmcv.dump(json_results, result_files['compas'])
        else:
            if isinstance(results[0], list):
                json_results = self._det2json(results)
                result_files['bbox'] = f'{outfile_prefix}.bbox.json'
                result_files['proposal'] = f'{outfile_prefix}.bbox.json'
                mmcv.dump(json_results, result_files['bbox']) 
            elif isinstance(results[0], tuple):
                json_results = self._segm2json(results)
                result_files['bbox'] = f'{outfile_prefix}.bbox.json'
                result_files['proposal'] = f'{outfile_prefix}.bbox.json'
                result_files['segm'] = f'{outfile_prefix}.segm.json'
                mmcv.dump(json_results[0], result_files['bbox'])
                mmcv.dump(json_results[1], result_files['segm'])
            elif isinstance(results[0], np.ndarray):
                json_results = self._proposal2json(results)
                result_files['proposal'] = f'{outfile_prefix}.proposal.json'
                mmcv.dump(json_results, result_files['proposal'])
            else:
                raise TypeError('invalid type of results')
        return result_files

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix, **kwargs)
        return result_files, tmp_dir