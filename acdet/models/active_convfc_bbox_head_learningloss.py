'''
Reference:
    https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmdet.models.builder import HEADS, build_loss
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared2FCBBoxHead
from mmdet.models.losses import accuracy

@HEADS.register_module()
class ActiveShared2FCBBoxLearningLossHead(Shared2FCBBoxHead):

    def __init__(self, *args, **kwargs):
        super(ActiveShared2FCBBoxLearningLossHead, self).__init__(
            *args,
            **kwargs)
        feat_channels = [256] * 4 # + [self.in_channels]
        feat_sizes = [256] * 4 # + [self.in_channels]
        # if self.num_shared_convs:
        #     feat_channels.extend([self.conv_out_channels] * self.num_shared_convs)
        #     feat_sizes.extend([self.conv_out_channels] * self.num_shared_convs)
        # if self.num_shared_fcs:
        #     feat_channels.extend([self.fc_out_channels] * self.num_shared_fcs)
        
        self.loss_GAP, self.loss_FC, self.loss_linear = self._add_loss_net(
            feature_sizes=feat_sizes,
            num_channels=feat_channels
            )
        self.drop_ll = False

    def _add_loss_net(self, feature_sizes=[32, 16, 8, 4],
                 num_channels=[64, 128, 256, 512],
                 interm_dim=128):
        loss_GAP = nn.ModuleList()
        for feature_size in feature_sizes:
            loss_GAP.append(nn.AdaptiveAvgPool2d((1, 1)))

        loss_FC = nn.ModuleList()
        for num_channel in num_channels:
            loss_FC.append(nn.Linear(num_channel, interm_dim))

        loss_linear = nn.Linear(len(num_channels) * interm_dim, 1)

        return loss_GAP, loss_FC, loss_linear

    def forward(self, feats, x):
        self.features = []
        self.features.extend(feats) # ([bs, 256, 184, 240], [2, 256, 92, 120], [2, 256, 46, 60], ...) 5 scales
        # self.features.append(x)
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
                # self.features.append(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
                # self.features.append(x)
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        if not self.drop_ll:
            self.learning_loss = self.forward_loss_net(self.features)

        return cls_score, bbox_pred

    def get_learning_loss(self):
        return self.learning_loss

    def forward_loss_net(self, features):
        outs = []
        for i in range(len(features)):
            if len(features[i].shape) >= 3:
                out = self.loss_GAP[i](features[i])
            out = out.view(out.size(0), -1)
            out = self.relu(self.loss_FC[i](out))
            outs.append(out)

        out = self.loss_linear(torch.cat(outs, 1))
        return out

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self, 
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             roi_splits=None,
             pos_inds_splits=None):
        if self.drop_ll:
            losses = super(ActiveShared2FCBBoxLearningLossHead, self).loss(
                        cls_score,
                        bbox_pred,
                        rois,
                        labels,
                        label_weights,
                        bbox_targets,
                        bbox_weights,
                        reduction_override=reduction_override)
            losses['loss_learning'] = torch.sum(losses['loss_cls'].new_zeros(1))
            return losses

        losses = super(ActiveShared2FCBBoxLearningLossHead, self).loss(
                    cls_score,
                    bbox_pred,
                    rois,
                    labels,
                    label_weights,
                    bbox_targets,
                    bbox_weights,
                    reduction_override='none')
        # get target loss
        loss_c = 0
        avg_factor_cls = max(torch.sum(label_weights > 0).float().item(), 1.)
        average_factor_reg = bbox_targets.size(0)
        if losses.get('loss_cls') is not None:
            loss_c = torch.stack([res.sum() for res in losses['loss_cls'].split(roi_splits)]) / avg_factor_cls
        loss_l = 0
        if losses.get('loss_bbox') is not None:
            loss_l = torch.stack([res.sum() for res in losses['loss_bbox'].split(pos_inds_splits)]) / average_factor_reg
        target_loss = loss_c + loss_l
        # target_loss = torch.sum(torch.stack([torch.stack(v.split(512)).mean(1) \
        #     for k, v in losses.items() if k.startswith('loss_')]), dim=0)
        # get lossnet loss
        pred_loss = self.get_learning_loss()
        # pred_loss = pred_loss.view(pred_loss.size(0))
        learning_loss = self.LossPredLoss(pred_loss, target_loss.reshape(-1, 1), margin=1.0)

        # losses = {k: v.mean() if k.startswith('loss_') else v for k, v in losses.items()}
        losses['loss_cls'] = losses['loss_cls'].sum() / avg_factor_cls
        losses['loss_bbox'] = losses['loss_bbox'].sum() / average_factor_reg
        losses['loss_learning'] = learning_loss * 0.3
        return losses


    # Loss Prediction Loss
    @force_fp32(apply_to=('input', 'target'))
    def LossPredLoss(self, input, target, margin=1.0, reduction='mean'):
        assert len(input) % 2 == 0, 'the batch size is not even.'
        assert input.shape == input.flip(0).shape

        input = (input - input.flip(0))[:len(input) // 2]
        # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target = (target - target.flip(0))[:len(target) // 2]
        target = target.detach()

        one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors

        if reduction == 'mean':
            loss = torch.sum(torch.clamp(margin - one * input, min=0))
            loss = loss / input.size(0)  # Note that the size of input is already halved
        elif reduction == 'none':
            loss = torch.clamp(margin - one * input, min=0)
        else:
            NotImplementedError()

        return loss