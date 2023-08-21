import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import ConvFCBBoxHead

@HEADS.register_module()
class ActiveShared2FCBBoxMIAODHead(ConvFCBBoxHead):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self, 
                 num_shared_convs=0,
                 num_shared_fcs=2,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(
            *args, 
            init_cfg=init_cfg, 
            **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        
        # add more loss func
        self.l_imgcls = nn.BCELoss()

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs_1, self.cls_fcs_1, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)
        self.cls_convs_2, self.cls_fcs_2, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)
        self.cls_convs_mil, self.cls_fcs_mil, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        
        # cleanup
        if self.with_reg:
            self.init_cfg = self.init_cfg[:-1]
            del self.fc_reg
        if self.with_cls:
            self.init_cfg = self.init_cfg[:-1]
            del self.fc_cls

        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                self.cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                self.cls_channels = self.num_classes + 1
            self.fc_cls_1 = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=self.cls_channels)
            self.fc_cls_2 = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=self.cls_channels)
            self.fc_cls_mil = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=self.cls_channels)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Normal', 
                    std=0.01, 
                    override=[
                        dict(name='fc_cls_1'),
                        dict(name='fc_cls_2'),
                        dict(name='fc_cls_mil'),
                        dict(name='fc_reg'),
                    ]),
                dict(
                    type='Xavier',
                    layer='Linear',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs_1'),
                        dict(name='cls_fcs_2'),
                        dict(name='cls_fcs_mil'),
                        dict(name='reg_fcs')
                    ])
            ]

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls_1 = x
        x_cls_2 = x
        x_cls_mil = x
        x_reg = x

        for conv in self.cls_convs_1:
            x_cls_1 = conv(x_cls_1)
        for conv in self.cls_convs_2:
            x_cls_2 = conv(x_cls_2)
        for conv in self.cls_convs_mil:
            x_cls_mil = conv(x_cls_mil)
        if x_cls_1.dim() > 2:
            if self.with_avg_pool:
                x_cls_1 = self.avg_pool(x_cls_1)
                x_cls_2 = self.avg_pool(x_cls_2)
                x_cls_mil = self.avg_pool(x_cls_mil)
            x_cls_1 = x_cls_1.flatten(1)
            x_cls_2 = x_cls_2.flatten(1)
            x_cls_mil = x_cls_mil.flatten(1)
        for fc in self.cls_fcs_1:
            x_cls_1 = self.relu(fc(x_cls_1))
        for fc in self.cls_fcs_2:
            x_cls_2 = self.relu(fc(x_cls_2))
        for fc in self.cls_fcs_mil:
            x_cls_mil = self.relu(fc(x_cls_mil))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        
        if self.with_cls:
            cls_score_1 = self.fc_cls_1(x_cls_1)
            cls_score_2 = self.fc_cls_2(x_cls_2)
            cls_score_mil = self.fc_cls_mil(x_cls_mil)
        else:
            raise NotImplementedError
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        y_head_cls_term2 = (cls_score_1 + cls_score_2) / 2
        y_head_cls_term2 = y_head_cls_term2.detach()    # (1024, 21)
        y_head_cls = cls_score_mil.softmax(1) * y_head_cls_term2.sigmoid().max(1, keepdim=True)[0].softmax(0)
        # originial implementation in RetinaNet only use foreground classes

        return [cls_score_1, cls_score_2, y_head_cls], bbox_pred

    def loss(self,
             list_cls_score,
             bbox_pred,
             rois,
             labels,
             *args, **kwargs):
        assert len(list_cls_score) == 3
        # Label set training
        cls_score_1, cls_score_2, y_head_cls = [i.float() for i in list_cls_score]
        bbox_pred = bbox_pred.float()

        loss_1 = self.loss_det(
            cls_score_1, bbox_pred, y_head_cls, rois, labels, *args, **kwargs
        )
        loss_2 = self.loss_det(
            cls_score_2, bbox_pred, y_head_cls, rois, labels, *args, **kwargs
        )
        loss_det_cls = (loss_1['loss_cls'] + loss_2['loss_cls']) / 2
        loss_det_loc = (loss_1['loss_bbox'] + loss_2['loss_bbox']) / 2
        loss_imgcls = (loss_1['loss_imgcls'] + loss_2['loss_imgcls']) / 2
        losses = dict(loss_cls=loss_det_cls, loss_bbox=loss_det_loc, loss_imgcls=loss_imgcls)
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'y_head_cls'))
    def loss_det(self, 
                 cls_score,
                 bbox_pred,
                 y_head_cls,
                 rois,
                 labels,
                 label_weights,
                 bbox_targets,
                 bbox_weights,
                 reduction_override=None,
                 **kwargs):
        losses = super(ActiveShared2FCBBoxMIAODHead, self).loss(
            cls_score,
            bbox_pred,
            rois,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            reduction_override=reduction_override)
        # mil loss
        labels_batch = cls_score.new_zeros(self.cls_channels)
        pos_inds = (labels >= 0) & (labels <= self.num_classes) # count bg as SSD
        labels_batch[labels[pos_inds].unique()] = 1

        y_head_cls_batch = y_head_cls.sum(0).clamp(1e-5, 1.0-1e-5)  # (#roi, C+1)
        l_imgcls = self.l_imgcls(y_head_cls_batch, labels_batch) * 0.1
        losses.update(dict(loss_imgcls=l_imgcls))
        return losses
        
    @force_fp32(apply_to=('cls_score_1', 'cls_score_2', 'y_head_cls'))
    def loss_wave_min(self,
                      cls_score_1,
                      cls_score_2, 
                      y_head_cls,):
        losses = self.loss_wave_dis(cls_score_1, cls_score_2, y_head_cls, False)
        
        y_pseudo = cls_score_1.new_zeros(self.cls_channels)
        # predict image pseudo label
        with torch.no_grad():
            y_pseudo = cls_score_1.sigmoid() + cls_score_2.sigmoid()
            y_pseudo = y_pseudo.max(0)[0] / 2
            y_pseudo[y_pseudo >= 0.5] = 1
            y_pseudo[y_pseudo < 0.5] = 0
        y_pseudo = y_pseudo.detach()
        # mil image score
        y_head_cls_batch = y_head_cls.sum(0).clamp(1e-5, 1.0-1e-5)  # (#roi, C+1)
        if y_pseudo.sum() == 0:  # ignore hard images
            l_imgcls = self.l_imgcls(y_head_cls_batch, y_pseudo) * 0
        else:
            l_imgcls = self.l_imgcls(y_head_cls_batch, y_pseudo) * 0.1
        losses.update(dict(unlabel_loss_imgcls=l_imgcls))
        return losses

    @force_fp32(apply_to=('cls_score_1', 'cls_score_2', 'y_head_cls'))
    def loss_wave_dis(self,
                      cls_score_1,
                      cls_score_2, 
                      y_head_cls,
                      minus):
        cls_score_1 = nn.Sigmoid()(cls_score_1)
        cls_score_2 = nn.Sigmoid()(cls_score_2)
        # mil weight
        w_i = y_head_cls.detach()
        diff = abs(cls_score_1 - cls_score_2)
        if minus:
            diff = 1 - diff
        l_det_cls_all = (diff * w_i).mean(dim=1).sum() * 0.5
        # self.param_lambda
        return dict(unlabel_loss_wavedis=l_det_cls_all)