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
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.sigmoid().contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


@HEADS.register_module()
class RoadOCC(BaseModule):
    r"""
    Args:
        in_channels (int): Channels of input feature.
        context_channels (int): Channels of transformed feature.
    """

    def __init__(
        self,
        in_channels=64,
        scale_factor=1,
        mask_thre = 0.4,
        norm_cfg=dict(type='BN3d', requires_grad=True),
    ):
        super(RoadOCC, self).__init__()
        # self.mask_net = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(in_channels//2),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels//2, 1, kernel_size=3, padding=1, stride=1),
        #     )
        
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, in_channels*2, kernel_size=3,
                      stride=1,padding=1, bias=False),
            build_norm_layer(norm_cfg, in_channels*2)[1],
            nn.ReLU(inplace=True),
        )
        # self.conv3d = nn.Conv3d(in_channels, in_channels*2, kernel_size=3, padding=1, stride=1)
        self.pred_conv = nn.Conv3d(in_channels*2, 1, kernel_size=1, padding=0, stride=1)
        
        
        # self.upsample = nn.Upsample(scale_factor = scale_factor , mode ='bilinear',align_corners = True)
        # self.dice_loss = build_loss(dict(type='CustomDiceLoss', use_sigmoid=True, loss_weight=1.)) # TODO: 定义dice loss
        self.dice_loss = BinaryDiceLoss()
        self.ce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.13]))  # From lss
        self.mask_thre = mask_thre

    def forward(self, input):
        """
            input (tensor): shape (N, C, H, W, Z)
        """
        # 先过一个3d卷积，再接一个pred_cov
        N,C,H,W,Z = input.shape
        mid_feature = self.conv3d(input) # (N, 2C, H, W, Z)
        road_mask = self.pred_conv(mid_feature) # (N, 1, H, W, Z)
        # road_mask = F.interpolate(road_mask, size=(H, W, 16))
        
        return road_mask
    
    
    def get_bev_mask_loss(self, gt_bev_mask, pred_bev_mask):
        bs, bev_h, bev_w = gt_bev_mask.shape
        b = gt_bev_mask.reshape(bs , bev_w * bev_h).permute(1, 0).to(torch.float)
        a = pred_bev_mask.reshape(bs, bev_w * bev_h).permute(1, 0)
        mask_ce_loss = self.ce_loss(a, b)
        mask_dice_loss = self.dice_loss(pred_bev_mask.reshape(bs, -1), gt_bev_mask.reshape(bs, -1))
        return dict(mask_ce_loss=mask_ce_loss, mask_dice_loss=mask_dice_loss)
    
    def get_road_mask_loss(self, gt_road_mask, pred_road_mask):
        bs, occ_h, occ_w, occ_z = gt_road_mask.shape
        pred_road_mask = F.interpolate(pred_road_mask, size=(occ_h, occ_w, occ_z))
        b = gt_road_mask.reshape(bs , occ_w * occ_h * occ_z).permute(1, 0).to(torch.float)
        a = pred_road_mask.reshape(bs, occ_w * occ_h * occ_z).permute(1, 0)
        mask_ce_loss = self.ce_loss(a, b)
        mask_dice_loss = self.dice_loss(pred_road_mask.reshape(bs, -1), gt_road_mask.reshape(bs, -1))
        return dict(mask_road_ce_loss=mask_ce_loss, mask_road_dice_loss=mask_dice_loss)
