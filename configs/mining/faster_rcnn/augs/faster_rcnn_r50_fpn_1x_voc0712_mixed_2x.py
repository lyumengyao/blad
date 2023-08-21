_base_ = './faster_rcnn_r50_fpn_1x_voc0712_mixed.py'

evaluation = dict(interval=5000)
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=25000)   # full is 25000, for 2*4 samples/iter
lr_config = dict(step=[17000, 23000])
checkpoint_config = dict(interval=5000, max_keep_ckpts=10)

fp16 = dict(loss_scale="dynamic")

log_config = dict(interval=100)