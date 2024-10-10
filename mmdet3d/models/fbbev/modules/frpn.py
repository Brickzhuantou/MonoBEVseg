# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint
from mmdet.models.backbones.resnet import BasicBlock
from mmdet.models import HEADS
import torch.utils.checkpoint as cp
from mmdet3d.models.builder import build_loss
from mmdet3d.models.fbbev.modules.road_occ import BinaryDiceLoss
from mmdet3d.models.dense_heads.map_head import SegmentationLoss


@HEADS.register_module()
class FRPN(BaseModule):
    r"""
    Args:
        in_channels (int): Channels of input feature.
        context_channels (int): Channels of transformed feature.
    """

    def __init__(
        self,
        in_channels=512,
        scale_factor=1,
        mask_thre = 0.4,
        binary_cls = True,
        multi_cls_num = 4,
        class_weights = [1.0, 1.0, 1.0, 1.0],
        semantic_thresh = 0.25,
    ):
        super(FRPN, self).__init__()
        self.binary_cls = binary_cls
        self.multi_cls_num = multi_cls_num
        self.semantic_thresh = semantic_thresh
        if self.binary_cls:
            self.mask_net = nn.Sequential(
                nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(in_channels//2),
                nn.ReLU(),
                nn.Conv2d(in_channels//2, 1, kernel_size=3, padding=1, stride=1),
                )
        else:
            self.mask_net = nn.Sequential(
                nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(in_channels//2),
                nn.ReLU(),
                nn.Conv2d(in_channels//2, self.multi_cls_num, kernel_size=1, padding=0, stride=1),
                # nn.Sigmoid()
                )
        self.upsample = nn.Upsample(scale_factor = scale_factor , mode ='bilinear',align_corners = True)

        # for binary cls
        self.dice_loss = BinaryDiceLoss()
        self.ce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.13]))  # From lss
        self.mask_thre = mask_thre
        
        # for multi cls
        self.semantic_seg_criterion = SegmentationLoss(
            class_weights=torch.tensor(class_weights).float(),
        )

    def forward(self, input):
        """
        """
        bev_mask = self.mask_net(input)            
        bev_mask = self.upsample(bev_mask)
        return bev_mask
    
    
    def get_bev_mask_loss(self, gt_bev_mask, pred_bev_mask):
        bs, bev_h, bev_w = gt_bev_mask.shape
        if self.binary_cls:
            b = gt_bev_mask.reshape(bs , bev_w * bev_h).permute(1, 0).to(torch.float)
            a = pred_bev_mask.reshape(bs, bev_w * bev_h).permute(1, 0)
            mask_ce_loss = self.ce_loss(a, b)
            mask_dice_loss = self.dice_loss(pred_bev_mask.reshape(bs, -1), gt_bev_mask.reshape(bs, -1))
            return dict(mask_frpn_ce_loss=mask_ce_loss, mask_frpn_dice_loss=mask_dice_loss)
        else:
            loss_seg_stage1 = self.semantic_seg_criterion(
                pred_bev_mask.unsqueeze(dim=1).float(),
                gt_bev_mask.unsqueeze(dim=1).long(),
            )
            return dict(seg_loss_LSS = loss_seg_stage1)


# DONE: 1. 标签制作为二值化的mask，只区分前景背景；2. 标签制作为多分类的mask，相当于提前做一遍语义分割，给former语义先验；
