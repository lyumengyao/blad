import json
import numpy as np
import cv2
import numpy as np
import json
import io
import itertools
from collections import defaultdict
import os.path as osp

from mmdet.datasets.api_wrappers import COCO as _COCO

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class COCO(_COCO):
    """ Inheret from mmdet COCO, which inhereted from pycocotools and changed interface
    """

    def get_len(self):
        img_labeled, img_unlabeled = 0, 0
        box_labeled, box_unlabeled = 0, 0
        for img_id in self.imgs.keys():
            anns = [ann for ann in self.imgToAnns[img_id] if ann['iscrowd'] == False]
            anns_labeled = [ann for ann in anns if ann['islabeled'] == True]
            if len(anns_labeled):
                img_labeled += 1
            else:
                img_unlabeled += 1
            box_labeled += len(anns_labeled)
            box_unlabeled += len(anns) - len(anns_labeled)
        return img_labeled, box_labeled, img_unlabeled, box_unlabeled

    def createIndex(self, quiet=False):
        # create index
        if not quiet:
            print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        imgid2idx, annsid2idx = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for i, ann in enumerate(self.dataset['annotations']):
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann
                annsid2idx[ann['id']] = i

        if 'images' in self.dataset:
            for i, img in enumerate(self.dataset['images']):
                imgs[img['id']] = img
                imgid2idx[img['id']] = i

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        if not quiet:
            print('index created!')

        # create class members
        self.anns = anns
        self.annsid2idx = annsid2idx
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.imgid2idx = imgid2idx 
        self.cats = cats


    def get_anns(self, imgIds=[], catIds=[], iscrowd=None, islabeled=None):
        """
        Get ann that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               islabeled (boolean)     : get anns for given crowd label (False or True)
               islabeled (boolean)     : get anns for given labeled label (False or True)
        :return: anns (dict array)     : dict array of ann
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            anns = self.dataset['annotations']
        else:
            if len(imgIds) == 0:
                anns = self.dataset['annotations']
            else:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
        if iscrowd is not None:
            anns = [ann for ann in anns if ann['iscrowd'] == iscrowd]
        if islabeled is not None:
            anns = [ann for ann in anns if ann['islabeled'] == islabeled]
        return anns
    
    def set_anns_labeled(self, annIds=[], return_unlabeled=False):
        annIds = annIds if _isArrayLike(annIds) else [annIds]
        labeled = []
        for annId in annIds:
            self.anns[annId]["islabeled"] = True
            img_id = self.anns[annId]["image_id"]
            self.dataset["images"][self.imgid2idx[img_id]]["partial"] = True
            labeled.append(len(self.get_anns(img_id, islabeled=False)))
        if return_unlabeled:
            return labeled

    def remove_image(self, imgIds=[]):
        if not _isArrayLike(imgIds):
            imgIds = [imgIds]
        imgIndex = sorted([self.imgid2idx[img_id] for img_id in imgIds if img_id in self.imgid2idx], reverse=True)
        annIndex = sorted([self.annsid2idx[i] for i in self.get_ann_ids(imgIds)], reverse=True)
        for img_idx in imgIndex:
            del self.dataset["images"][img_idx]
        for ann_idx in annIndex:
            del self.dataset["annotations"][ann_idx]
        self.createIndex(quiet=True)

    def add_image(self, images=[], anns=[]):
        if not _isArrayLike(images):
            images = [images]
            anns = [anns]
        else:
            assert len(images) == len(anns)
        for imgs, ans in zip(images, anns):
            self.dataset["images"].append(imgs)
            for a in ans:
                assert a["islabeled"]   # check validity
            self.dataset["annotations"].extend(ans)
        self.createIndex(quiet=True)

def coco_load_obj(path, filter_unlabeled=True, config=None):
    coco = COCO(path)
    return coco, {'categories': coco.dataset['categories']}

def dump_json(save_path, dict_content):
    file = open(save_path, 'w')
    file.write(json.dumps(dict_content))
    file.close()

def split_partial(image_set, meta):
    partial, missing, unlabeled = {}, {}, {}
    partial.update(meta)
    missing.update(meta)
    unlabeled.update(meta)
    pimages, uimages = [], []
    pannotations, uannotations = [], []

    for i, img in enumerate(image_set.dataset['images']):
        if img["partial"]:
            pimages.append(img)
            pannotations.append(img["id"])
        else:
            uimages.append(img)
            uannotations.append(img["id"])
    
    partial['images'] = pimages
    partial['annotations'] = image_set.get_anns(pannotations, islabeled=True) # only save labeled annos in partial.json
    missing['images'] = pimages
    missing['annotations'] = image_set.get_anns(pannotations, islabeled=False) # only save labeled annos in partial.json
    unlabeled['images'] = uimages
    unlabeled['annotations'] = image_set.get_anns(uannotations)
    return partial, missing, unlabeled

def coco_save_obj(image_set, meta, save_path, labeled=None, count=-1):
    dump_json(save_path, image_set.dataset)
    if not labeled:
        # split partial and unlabeled
        partial, missing, unlabeled = split_partial(image_set, meta)
        work_dir = "/".join(save_path.split("/")[:-1])
        dump_json(osp.join(work_dir, "partial.json"), partial)
        dump_json(osp.join(work_dir, "missing.json"), missing)
        dump_json(osp.join(work_dir, "unlabeled.json"), unlabeled)

def coco_load(path, filter_unlabeled=True, config=None):
    cntbox = config.get('count_box', False)
    image_set = []
    coco_ann = json.load(open(path))
    meta = {'categories': coco_ann['categories']}
    images = coco_ann['images']

    imageid2anno = {}
    imageid2info = {}
    imageid2boxcnt = {}
    for image in images:
        imageid2info[image['id']] = image
        imageid2anno[image['id']] = []
        imageid2boxcnt[image['id']] = 0

    annotations = coco_ann['annotations']
    for anno in annotations:
        # filter pseudo label from last step
        if anno.get("is_pseudo_label", False):
            continue
        image_id = anno['image_id']
        if image_id in imageid2anno:
            imageid2anno[image_id].append(anno)
            if cntbox:
                img_info = imageid2info[image_id]
                if min(img_info['width'], img_info['height']) < 32 or anno['iscrowd']:
                    continue  # if ommited during training, keep in instances but omit the count
                imageid2boxcnt[image_id] += 1
    
    for index in range(len(images)):
        # filter images with empty bbox
        image_id = images[index]['id']
        if filter_unlabeled and len(imageid2anno[image_id]) == 0:
            continue
        if cntbox and imageid2boxcnt[image_id] == 0:
            continue
        tmp = {}
        tmp['info'] = imageid2info[image_id]
        tmp['instances'] = imageid2anno[image_id]
        if cntbox: tmp['cntbox'] = imageid2boxcnt[image_id]
        image_set.append(tmp)
    return image_set, meta

def coco_combine(image_sets):
    combined_set = []
    for image_set in image_sets:
        combined_set += image_set
    return combined_set
        

def coco_save(image_set, meta, save_path, labeled=None, count=-1):
    coco_anno = {}
    coco_anno.update(meta)
    images = []
    annotations = []
    for index in range(len(image_set)):
        img_info = image_set[index]['info']
        if labeled is not None and not labeled:
            img_info["partial"] = False
        images.append(img_info)
        if labeled is not None:
            for ann in image_set[index]['instances']:
                ann["islabeled"] = True if labeled else False   # only when ratio==0
        annotations += image_set[index]['instances']
        if count > 0 and index == count:
            save_path2 = '/'.join(save_path.split('/')[:-1]) + '/unlabeled_selection.json'
            print(f"Saving {save_path2}")
            coco_anno['images'] = images
            coco_anno['annotations'] = annotations
            dump_json(save_path2, coco_anno)

    if count > 0 and index < count:
        save_path2 = '/'.join(save_path.split('/')[:-1]) + '/unlabeled_selection.json'
        print(f"Saving {save_path2}")
        coco_anno['images'] = images
        coco_anno['annotations'] = annotations
        dump_json(save_path2, coco_anno)
    coco_anno['images'] = images
    coco_anno['annotations'] = annotations

    dump_json(save_path, coco_anno)
