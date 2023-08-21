import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmcv.cnn import ConvModule

from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import build_linear_layer
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import ConvFCBBoxHead
from mmdet.models.losses import accuracy
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss

def Gaussian(y, mu, var):
    eps = 0.3
    result = (y-mu)/var
    result = (result**2)/2*(-1)
    exp = torch.exp(result)
    result = exp/(math.sqrt(2*math.pi))/(var + eps)
    return result

@mmcv.jit(derivate=True, coderize=True)
def NLL_loss(bbox_pred, bbox_gt, var):
    bbox_var = torch.sigmoid(var)
    prob = Gaussian(bbox_gt, bbox_pred, bbox_var)
    return prob

@LOSSES.register_module()
class NLLLoss(nn.Module):
    def __init__(self, reduction=None, loss_weight=1.0):
        super(NLLLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                var=None,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override == 'none' and avg_factor is None
        assert var is not None
        loss_bbox = self.loss_weight * NLL_loss(pred, target, var)
        # if weight is specified, apply element-wise weight
        if weight is not None:
            loss_bbox = loss_bbox * weight
        return loss_bbox

@HEADS.register_module()
class ActiveShared2FCBBoxALMDNHead(ConvFCBBoxHead):
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
                 num_gmms=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 loss_bbox=dict(
                     type='NLL_loss', loss_weight=1.0),
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(
            *args, 
            loss_bbox=loss_bbox, 
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
        self.num_gmms = num_gmms

        self.relu = nn.ReLU(inplace=True)

        if self.with_reg:
            self.init_cfg = self.init_cfg[:-1]
            del self.fc_reg
        if self.with_cls:
            self.init_cfg = self.init_cfg[:-1]
            del self.fc_cls

        ini_Xavier, ini_normal = [], []
        for head_i in range(self.num_gmms):
            shared_convs, shared_fcs, \
            cls_convs, cls_fcs, \
            reg_convs, reg_fcs, \
            fc_cls_mu, fc_cls_var, fc_cls_pi, \
            fc_reg_mu, fc_reg_var, fc_reg_pi = \
                self._add_cls_reg_heads()
            self.add_module(f'shared_convs_{head_i}', shared_convs)
            self.add_module(f'shared_fcs_{head_i}', shared_fcs)
            self.add_module(f'cls_convs_{head_i}', cls_convs)
            self.add_module(f'cls_fcs_{head_i}', cls_fcs)
            self.add_module(f'reg_convs_{head_i}', reg_convs)
            self.add_module(f'reg_fcs_{head_i}', reg_fcs)
            self.add_module(f'fc_cls_mu{head_i}', fc_cls_mu)
            self.add_module(f'fc_cls_var{head_i}', fc_cls_var)
            self.add_module(f'fc_cls_pi{head_i}', fc_cls_pi)
            self.add_module(f'fc_reg_mu{head_i}', fc_reg_mu)
            self.add_module(f'fc_reg_var{head_i}', fc_reg_var)
            self.add_module(f'fc_reg_pi{head_i}', fc_reg_pi)
            ini_normal.extend([
                dict(name=f'fc_cls_mu{head_i}'),
                dict(name=f'fc_cls_var{head_i}'),
                dict(name=f'fc_cls_pi{head_i}'),
                dict(name=f'fc_reg_mu{head_i}'),
                dict(name=f'fc_reg_var{head_i}'),
                dict(name=f'fc_reg_pi{head_i}'),
            ])
            ini_Xavier.extend([
                dict(name=f'shared_fcs_{head_i}'),
                dict(name=f'cls_fcs_{head_i}'),
                dict(name=f'reg_fcs_{head_i}')]
            )
        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Normal',
                    std=0.01, 
                    override=ini_normal),
                dict(
                    type='Xavier',
                    layer='Linear',
                    override=ini_Xavier)
            ]

        self.active = False

    def _add_cls_reg_heads(self, ):

        # add shared convs and fcs
        shared_convs, shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        cls_convs, cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        reg_convs, reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            fc_cls_mu = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
            fc_cls_var = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
            fc_cls_pi = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=1)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                        self.num_classes)
            fc_reg_mu = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)
            fc_reg_var = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)
            fc_reg_pi = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        return shared_convs, shared_fcs, \
               cls_convs, cls_fcs, \
               reg_convs, reg_fcs, \
               fc_cls_mu, fc_cls_var, fc_cls_pi, \
               fc_reg_mu, fc_reg_var, fc_reg_pi

    def forward(self, x):
        cls_scores, bbox_preds = [], []
        for head_i in range(self.num_gmms):
            cls_score, bbox_pred = self.forward_single(x, head_i)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return cls_scores, bbox_preds
    
    def forward_single(self, x, head_i):
        # shared part
        if self.num_shared_convs > 0:
            shared_convs = getattr(self, f'shared_convs_{head_i}')
            for conv in shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            shared_fcs = getattr(self, f'shared_fcs_{head_i}')
            for fc in shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        cls_convs = getattr(self, f'cls_convs_{head_i}')
        for conv in cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        cls_fcs = getattr(self, f'cls_fcs_{head_i}')
        for fc in cls_fcs:
            x_cls = self.relu(fc(x_cls))

        reg_convs = getattr(self, f'reg_convs_{head_i}')
        for conv in reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        reg_fcs = getattr(self, f'reg_fcs_{head_i}')
        for fc in reg_fcs:
            x_reg = self.relu(fc(x_reg))

        fc_cls_mu = getattr(self, f'fc_cls_mu{head_i}')
        cls_score_mu = fc_cls_mu(x_cls) if self.with_cls else None
        fc_cls_var = getattr(self, f'fc_cls_var{head_i}')
        cls_score_var = fc_cls_var(x_cls) if self.with_cls else None
        fc_cls_pi = getattr(self, f'fc_cls_pi{head_i}')
        cls_score_pi = fc_cls_pi(x_cls) if self.with_cls else None

        fc_reg_mu = getattr(self, f'fc_reg_mu{head_i}')
        bbox_pred_mu = fc_reg_mu(x_reg) if self.with_reg else None
        fc_reg_var = getattr(self, f'fc_reg_var{head_i}')
        bbox_pred_var = fc_reg_var(x_reg) if self.with_reg else None
        fc_reg_pi = getattr(self, f'fc_reg_pi{head_i}')
        bbox_pred_pi = fc_reg_pi(x_reg) if self.with_reg else None
        return [cls_score_mu, cls_score_var, cls_score_pi], \
            [bbox_pred_mu, bbox_pred_var, bbox_pred_pi]

    def loss(self,
             list_cls_score,
             list_bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override='none'):
        losses = dict()
        assert self.num_gmms == len(list_cls_score) == len(list_bbox_pred)
        list_cls_score = [[j.float() for j in i] for i in list_cls_score]
        list_bbox_pred = [[j.float() for j in i] for i in list_bbox_pred]
        
        def get_pis(list_pis):
            pis = torch.stack([pi.reshape(-1) for pi in list_pis]).transpose(0, 1)
            pis = (torch.softmax(pis, dim=1)).transpose(0, 1).reshape(-1)
            pis = torch.split(pis, list_pis[0].reshape(-1).size(0), dim=0)
            return pis

        losses_cls = []
        cls_pis = []
        for head_i, cls_score in enumerate(list_cls_score):
            cls_score_mu, cls_score_var, cls_score_pi = cls_score
            if cls_score_mu is not None:
                cls_score_var = torch.sigmoid(cls_score_var)
                rand_val = torch.randn_like(cls_score_var)
                cls_score = (cls_score_mu + torch.sqrt(cls_score_var) * rand_val).view(-1, self.num_classes + 1)
                if cls_score.numel() > 0:
                    loss_cls_ = self.loss_cls(
                        cls_score,
                        labels,
                        label_weights,
                        reduction_override='none')
                    losses_cls.append(loss_cls_)
                    # if self.custom_activation:
                    #     acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    #     losses.update(acc_)
                    # else:
                    #     losses['acc'] = accuracy(cls_score, labels)
                    cls_pis.append(cls_score_pi)
        if len(cls_pis):
            cls_pis = torch.stack(get_pis(cls_pis))
            losses_cls = torch.sum(cls_pis * torch.stack(losses_cls), dim=0)
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = losses_cls.sum() / avg_factor
        
        losses_bbox = []
        reg_pis = []
        for head_i, bbox_pred in enumerate(list_bbox_pred):
            bbox_pred_mu, bbox_pred_var, bbox_pred_pi = bbox_pred
            if bbox_pred_mu is not None:
                bg_class_ind = self.num_classes
                # 0~self.num_classes-1 are FG, self.num_classes is BG
                pos_inds = (labels >= 0) & (labels < bg_class_ind)
                # do not perform bounding box regression for BG anymore.
                if pos_inds.any():
                    if self.reg_decoded_bbox:
                        # When the regression loss (e.g. `IouLoss`,
                        # `GIouLoss`, `DIouLoss`) is applied directly on
                        # the decoded bounding boxes, it decodes the
                        # already encoded coordinates to absolute format.
                        bbox_pred_mu = self.bbox_coder.decode(rois[:, 1:], bbox_pred_mu)
                    if self.reg_class_agnostic:
                        pos_bbox_pred_mu = bbox_pred_mu.view(
                            bbox_pred_mu.size(0), 4)[pos_inds.type(torch.bool)]
                        pos_bbox_pred_var = bbox_pred_var.view(
                            bbox_pred_var.size(0), 4)[pos_inds.type(torch.bool)]
                        pos_bbox_pred_pi = bbox_pred_pi.view(
                            bbox_pred_pi.size(0), 4)[pos_inds.type(torch.bool)]
                    else:
                        pos_bbox_pred_mu = bbox_pred_mu.view(
                            bbox_pred_mu.size(0), -1,
                            4)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]
                        pos_bbox_pred_var = bbox_pred_var.view(
                            bbox_pred_var.size(0), -1,
                            4)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]
                        pos_bbox_pred_pi = bbox_pred_pi.view(
                            bbox_pred_pi.size(0), -1,
                            4)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]
                    losses_bbox.append(
                        self.loss_bbox(
                            pos_bbox_pred_mu,
                            bbox_targets[pos_inds.type(torch.bool)],
                            var=pos_bbox_pred_var,
                            weight=bbox_weights[pos_inds.type(torch.bool)],
                            reduction_override='none'))
                    reg_pis.append(pos_bbox_pred_pi)
        if len(reg_pis):
            reg_pis = get_pis(reg_pis)
            reg_pis = torch.stack([pi.view(-1, 4) for pi in reg_pis])
            losses_bbox = torch.sum(reg_pis * torch.stack(losses_bbox), dim=0)
            epsi = 10**-9
            # balance parameter
            balance = 2.0
            losses_bbox = -torch.log(losses_bbox + epsi)/balance
            avg_factor = bbox_targets.size(0)
            losses['loss_bbox'] = losses_bbox.sum() / avg_factor
        else:
            losses['loss_bbox'] = list_bbox_pred[0][0][pos_inds].sum()
        return losses

    def get_bboxes(self,
                   rois,
                   list_cls_score,
                   list_bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                Fisrt tensor is `det_bboxes`, has the shape
                (num_boxes, 5) and last
                dimension 5 represent (tl_x, tl_y, br_x, br_y, score).
                Second tensor is the labels with shape (num_boxes, ).
        """
        # force fp32
        list_cls_score = [[j.float() for j in i] for i in list_cls_score]
        list_bbox_pred = [[j.float() for j in i] for i in list_bbox_pred]

        # handle multi-heads
        def get_pis(list_pis):
            pis = torch.stack([pi.reshape(-1) for pi in list_pis]).transpose(0, 1)
            pis = (torch.softmax(pis, dim=1)).transpose(0, 1).reshape(-1)
            pis = torch.split(pis, list_pis[0].reshape(-1).size(0), dim=0)
            return pis
        
        reg_pis = get_pis([res[-1] for res in list_bbox_pred])
        for head_i, bbox_pred_i in enumerate(list_bbox_pred):
            bbox_pred_i[1] = torch.sigmoid(bbox_pred_i[1])
            bbox_pred_i[2] = reg_pis[head_i].view(-1, 4 * self.num_classes)

        cls_pis = get_pis([res[-1] for res in list_cls_score])
        for head_i, cls_score_i in enumerate(list_cls_score):
            cls_score_i[1] = torch.sigmoid(cls_score_i[1])
            cls_score_i[2] = cls_pis[head_i].view(-1, 1)

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            raise NotImplementedError
            scores = self.loss_cls.get_activation(cls_score)
        else:
            for cls_score_i in list_cls_score:
                cls_score_i[0] = F.softmax(cls_score_i[0], dim=-1)
            scores = torch.sum(torch.stack([cls_score_i[0] * cls_score_i[2] for cls_score_i in list_cls_score]), dim=0)
            if getattr(self, "active"):
                cls_al_un = torch.sum(torch.stack([cls_score_i[1] * cls_score_i[2] for cls_score_i in list_cls_score]), dim=0)
                cls_ep_un = torch.sum(torch.stack(
                    [(cls_score_i[0] - scores)**2 * cls_score_i[2] for cls_score_i in list_cls_score]
                    ), dim=0)
            
        bbox_pred = torch.sum(torch.stack([bbox_pred_i[0] * bbox_pred_i[2] for bbox_pred_i in  list_bbox_pred]), dim=0)
        if getattr(self, "active"):
            reg_al_un = torch.sum(torch.stack([bbox_pred_i[1] * bbox_pred_i[2] for bbox_pred_i in  list_bbox_pred]), dim=0)
            reg_ep_un = torch.sum(torch.stack(
                [(bbox_pred_i[0] - bbox_pred)**2 * bbox_pred_i[2] for bbox_pred_i in  list_bbox_pred]
                ), dim=0)
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:

            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels, inds = multiclass_nms(bboxes, scores,
                                                          cfg.score_thr, cfg.nms,
                                                          cfg.max_per_img,
                                                          return_inds=True)
            if getattr(self, "active"):
                cls_al_un = cls_al_un[:, :-1].reshape(-1)[inds].unsqueeze(1) # (1K, 21)
                cls_ep_un = cls_ep_un[:, :-1].reshape(-1)[inds].unsqueeze(1) # (1K, 21)
                reg_al_un = reg_al_un.reshape(-1, 4)[inds] # (1K, 80)
                reg_ep_un = reg_ep_un.reshape(-1, 4)[inds] # (1K, 80)
                det_bboxes = torch.cat((det_bboxes, cls_al_un, cls_ep_un, reg_al_un, reg_ep_un), dim=1)

            return det_bboxes, det_labels
