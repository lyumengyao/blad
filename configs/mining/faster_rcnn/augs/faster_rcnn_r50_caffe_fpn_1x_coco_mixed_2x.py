_base_ = 'faster_rcnn_r50_caffe_fpn_1x_coco_mixed.py'

evaluation = dict(interval=36000)
runner = dict(max_iters=180000)   # full is 25000, for 2*4 samples/iter
lr_config = dict(step=[120000, 160000])
checkpoint_config = dict(interval=36000, max_keep_ckpts=10)