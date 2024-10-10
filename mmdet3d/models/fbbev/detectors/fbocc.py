# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import torch
import torch.nn.functional as F
import torch.nn as nn
from mmcv.runner import force_fp32
import os
from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors import CenterPoint
from mmdet3d.models.builder import build_head, build_neck
import numpy as np
import copy 
# import spconv.pytorch as spconv
from tqdm import tqdm 
from mmdet3d.models.fbbev.utils import run_time
import torch
from torchvision.utils import make_grid
import torchvision
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
from mmcv.runner import get_dist_info
from mmdet.core import reduce_mean
import mmcv
# from mmdet3d.datasets.utils import nuscenes_get_rt_matrix
from mmdet3d.core.bbox import box_np_ops # , corner_to_surfaces_3d, points_in_convex_polygon_3d_jit



def generate_forward_transformation_matrix(bda, img_meta_dict=None):
    b = bda.size(0)
    hom_res = torch.eye(4)[None].repeat(b, 1, 1).to(bda.device)
    for i in range(b):
        hom_res[i, :3, :3] = bda[i]
    return hom_res


@DETECTORS.register_module()
class FBOCC(CenterPoint):

    def __init__(self, 
                 # BEVDet components
                 forward_projection=None,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 # BEVFormer components
                 backward_projection=None,
                 # FB-BEV components
                 frpn=None,
                 # depth_net
                 depth_net=None,
                 # occupancy head
                 occupancy_head=None,
                 # other settings.
                 use_depth_supervision=False,
                 readd=False,
                 fix_void=False,
                 occupancy_save_path=None,
                 do_history=True,
                 interpolation_mode='bilinear',
                 history_cat_num=16,
                 history_cat_conv_out_channels=None,
                 single_bev_num_channels=80,

                  **kwargs):
        super(FBOCC, self).__init__(**kwargs)
        self.fix_void = fix_void
      
        # BEVDet init
        self.forward_projection = builder.build_neck(forward_projection) if forward_projection else None
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone) if img_bev_encoder_backbone else None
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck) if img_bev_encoder_neck else None

        # BEVFormer init
        self.backward_projection = builder.build_head(backward_projection) if backward_projection else None
    
        # FB-BEV init
        if not self.forward_projection: assert not frpn, 'frpn relies on LSS'
        self.frpn = builder.build_head(frpn) if frpn else None

        # Depth Net
        self.depth_net = builder.build_head(depth_net) if depth_net else None

        # Occupancy Head
        self.occupancy_head = builder.build_head(occupancy_head) if occupancy_head else None

        
        self.readd = readd # fuse voxel features and bev features
        
        self.use_depth_supervision = use_depth_supervision
        
        self.occupancy_save_path = occupancy_save_path # for saving data\for submitting to test server

        # Deal with history
        self.single_bev_num_channels = single_bev_num_channels
        self.do_history = do_history
        self.interpolation_mode = interpolation_mode
        self.history_cat_num = history_cat_num
        self.history_cam_sweep_freq = 0.5 # seconds between each frame
        history_cat_conv_out_channels = (history_cat_conv_out_channels 
                                         if history_cat_conv_out_channels is not None 
                                         else self.single_bev_num_channels)
        ## Embed each sample with its relative temporal offset with current timestep
        # conv = nn.Conv2d if self.forward_projection.nx[-1] == 1 else nn.Conv3d
        # self.history_keyframe_time_conv = nn.Sequential(
        #      conv(self.single_bev_num_channels + 1,
        #              self.single_bev_num_channels,
        #              kernel_size=1,
        #              padding=0,
        #              stride=1),
        #      nn.SyncBatchNorm(self.single_bev_num_channels),
        #      nn.ReLU(inplace=True))
        # ## Then concatenate and send them through an MLP.
        # self.history_keyframe_cat_conv = nn.Sequential(
        #     conv(self.single_bev_num_channels * (self.history_cat_num + 1),
        #             history_cat_conv_out_channels,
        #             kernel_size=1,
        #             padding=0,
        #             stride=1),
        #     nn.SyncBatchNorm(history_cat_conv_out_channels),
        #     nn.ReLU(inplace=True))
        self.history_sweep_time = None
        self.history_bev = None
        self.history_bev_before_encoder = None
        self.history_seq_ids = None
        self.history_forward_augs = None
        self.count = 0

    def with_specific_component(self, component_name):
        """Whether the model owns a specific component"""
        return getattr(self, component_name, None) is not None
    
    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()
        
        rot = sensor2egos[:,:,:3,:3]
        tran = sensor2egos[:,:,:3,3]
        return [imgs, rot, tran, intrins,
                post_rots, post_trans, bda]
    
    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
      
        x = self.img_backbone(imgs)
       
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
      
        return x

    @force_fp32()
    def bev_encoder(self, x):
        if self.with_specific_component('img_bev_encoder_backbone'):
            x = self.img_bev_encoder_backbone(x)
        
        if self.with_specific_component('img_bev_encoder_neck'):
            x = self.img_bev_encoder_neck(x)
        
        if type(x) not in [list, tuple]:
             x = [x]

        return x
        
    @force_fp32()
    def fuse_history(self, curr_bev, img_metas, bda): # align features with 3d shift

        voxel_feat = True  if len(curr_bev.shape) == 5 else False
        if voxel_feat:
            curr_bev = curr_bev.permute(0, 1, 4, 2, 3) # n, c, z, h, w
        
        seq_ids = torch.LongTensor([
            single_img_metas['sequence_group_idx'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        forward_augs = generate_forward_transformation_matrix(bda)

        curr_to_prev_ego_rt = torch.stack([
            single_img_metas['curr_to_prev_ego_rt']
            for single_img_metas in img_metas]).to(curr_bev)

        ## Deal with first batch
        if self.history_bev is None:
            self.history_bev = curr_bev.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_forward_augs = forward_augs.clone()

            # Repeat the first frame feature to be history
            if voxel_feat:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1, 1) 
            else:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1)
            # All 0s, representing current timestep.
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_cat_num)


        self.history_bev = self.history_bev.detach()

        assert self.history_bev.dtype == torch.float32

        ## Deal with the new sequences
        # First, sanity check. For every non-start of sequence, history id and seq id should be same.

        assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
                "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)

        ## Replace all the new sequences' positions in history with the curr_bev information
        self.history_sweep_time += 1 # new timestep, everything in history gets pushed back one.
        if start_of_sequence.sum()>0:
            if voxel_feat:    
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1, 1)
            else:
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1)
            
            self.history_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]


        ## Get grid idxs & grid2bev first.
        if voxel_feat:
            n, c_, z, h, w = curr_bev.shape

        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack(
            (xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h, w, z, 4, 1)

        # This converts BEV indices to meters
        # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
        # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
        feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.forward_projection.dx[0]
        feat2bev[1, 1] = self.forward_projection.dx[1]
        feat2bev[2, 2] = self.forward_projection.dx[2]
        feat2bev[0, 3] = self.forward_projection.bx[0] - self.forward_projection.dx[0] / 2.
        feat2bev[1, 3] = self.forward_projection.bx[1] - self.forward_projection.dx[1] / 2.
        feat2bev[2, 3] = self.forward_projection.bx[2] - self.forward_projection.dx[2] / 2.
        # feat2bev[2, 2] = 1
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1,4,4)

        ## Get flow for grid sampling.
        # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
        # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
        # transform to previous grid locations.
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt
                   @ torch.inverse(forward_augs) @ feat2bev)

        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid

        # normalize and sample
        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
        grid = grid[:,:,:,:, :3,0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0
        

        tmp_bev = self.history_bev
        if voxel_feat: 
            n, mc, z, h, w = tmp_bev.shape
            tmp_bev = tmp_bev.reshape(n, mc, z, h, w)
        sampled_history_bev = F.grid_sample(tmp_bev, grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4), align_corners=True, mode=self.interpolation_mode)

        ## Update history
        # Add in current frame to features & timestep
        self.history_sweep_time = torch.cat(
            [self.history_sweep_time.new_zeros(self.history_sweep_time.shape[0], 1), self.history_sweep_time],
            dim=1) # B x (1 + T)

        if voxel_feat:
            sampled_history_bev = sampled_history_bev.reshape(n, mc, z, h, w)
            curr_bev = curr_bev.reshape(n, c_, z, h, w)
        feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=1) # B x (1 + T) * 80 x H x W or B x (1 + T) * 80 xZ x H x W 

        # Reshape and concatenate features and timestep
        feats_to_return = feats_cat.reshape(
                feats_cat.shape[0], self.history_cat_num + 1, self.single_bev_num_channels, *feats_cat.shape[2:]) # B x (1 + T) x 80 x H x W
        if voxel_feat:
            feats_to_return = torch.cat(
            [feats_to_return, self.history_sweep_time[:, :, None, None, None, None].repeat(
                1, 1, 1, *feats_to_return.shape[3:]) * self.history_cam_sweep_freq
            ], dim=2) # B x (1 + T) x 81 x Z x H x W
        else:
            feats_to_return = torch.cat(
            [feats_to_return, self.history_sweep_time[:, :, None, None, None].repeat(
                1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
            ], dim=2) # B x (1 + T) x 81 x H x W

        # Time conv
        feats_to_return = self.history_keyframe_time_conv(
            feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (1 + T) x 80 xZ x H x W

        # Cat keyframes & conv
        feats_to_return = self.history_keyframe_cat_conv(
            feats_to_return.reshape(
                feats_to_return.shape[0], -1, *feats_to_return.shape[3:])) # B x C x H x W or B x C x Z x H x W
        
        self.history_bev = feats_cat[:, :-self.single_bev_num_channels, ...].detach().clone()
        self.history_sweep_time = self.history_sweep_time[:, :-1]
        self.history_forward_augs = forward_augs.clone()
        if voxel_feat:
            feats_to_return = feats_to_return.permute(0, 1, 3, 4, 2)
        if not self.do_history:
            self.history_bev = None
        return feats_to_return.clone()


    def extract_img_bev_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""

        return_map = {}

        context = self.image_encoder(img[0])
        cam_params = img[1:7]
        if self.with_specific_component('depth_net'):
            mlp_input = self.depth_net.get_mlp_input(*cam_params)
            context, depth = self.depth_net(context, mlp_input)
            return_map['depth'] = depth
            return_map['context'] = context
        else:
            context=None
            depth=None
        
        if self.with_specific_component('forward_projection'):
            bev_feat = self.forward_projection(cam_params, context, depth, **kwargs)
            return_map['cam_params'] = cam_params
        else:
            bev_feat = None
        
        if self.with_specific_component('frpn'): # not used in FB-OCC
            bev_mask_logit = self.frpn(bev_feat)
            bev_mask = bev_mask_logit.sigmoid() > self.frpn.mask_thre
            
            if bev_mask.requires_grad: # during training phase
                gt_bev_mask = kwargs['gt_bev_mask'].to(torch.bool)
                bev_mask = gt_bev_mask | bev_mask
            return_map['bev_mask_logit'] = bev_mask_logit    
        else:
            bev_mask = None

        if self.with_specific_component('backward_projection'):
            
            if bev_feat == None:
                lss_bev = None
            else:
                lss_bev = bev_feat.mean(-1)
            
            bev_feat_refined = self.backward_projection([context],
                                        img_metas,
                                        lss_bev=lss_bev,
                                        cam_params=cam_params,
                                        bev_mask=bev_mask,
                                        gt_bboxes_3d=None, # debug
                                        pred_img_depth=depth)
            
            
            if self.backward_projection.transformer.encoder.return_intermediate==True:
                # 输出多层bev特征
                for i in range(len(bev_feat_refined)):
                    if self.readd:
                        bevfeat_add = (bev_feat_refined[i][..., None] + bev_feat).mean(-1)
                        bev_feat = self.bev_encoder(bevfeat_add)
                        return_map['img_bev_feat_{}'.format(i)] = bev_feat
                    else:
                        bevfeat_add = bev_feat_refined[i]       
                        bev_feat = self.bev_encoder(bevfeat_add)
                        return_map['img_bev_feat_{}'.format(i)] = bev_feat
            else:
                bev_feat = self.bev_encoder(bev_feat_refined)
                return_map['img_bev_feat'] = bev_feat
                                    
        else:
            # 单层transformer
            bev_feat = bev_feat.mean(-1)
            # Fuse History
            # bev_feat = self.fuse_history(bev_feat, img_metas, img[6])
            bev_feat = self.bev_encoder(bev_feat)
            return_map['img_bev_feat'] = bev_feat

        return return_map

    def extract_lidar_bev_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""

        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        bev_feat = self.pts_middle_encoder(voxel_features, coors, batch_size)
        bev_feat = self.bev_encoder(bev_feat)
        return bev_feat

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        results={}
        if img is not None and self.with_specific_component('image_encoder'):
            img = self.prepare_inputs(img)
            results.update(self.extract_img_bev_feat(img, img_metas, **kwargs))
        if points is not None and self.with_specific_component('pts_voxel_encoder'):
            results['lidar_bev_feat'] = self.extract_lidar_bev_feat(points, img, img_metas)

        return results
    
    def forward_pts_train(self,
                        pts_feats,
                        gt_bboxes_3d,
                        gt_labels_3d,
                        semantic_indices,
                        img_metas,
                        gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [semantic_indices, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      semantic_indices=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      gt_occupancy_flow=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """


        results= self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()

        if  self.with_pts_bbox:
            
            
            if self.backward_projection and self.backward_projection.transformer.encoder.return_intermediate==True:
                # 采用多层bevformer
                for i in range(self.backward_projection.transformer.encoder.num_layers):
                    losses_pts = self.forward_pts_train(results['img_bev_feat_{}'.format(i)], gt_bboxes_3d,
                                            gt_labels_3d, semantic_indices, img_metas,
                                            gt_bboxes_ignore)
                    loss_dict = dict()
                    for key, value in losses_pts.items():
                        loss_dict['img_bev_feat_{}_{}'.format(i, key)] = value
                    losses.update(loss_dict)            
            else:
                losses_pts = self.forward_pts_train(results['img_bev_feat'], gt_bboxes_3d,
                                                gt_labels_3d, semantic_indices, img_metas,
                                                gt_bboxes_ignore)
                
                losses.update(losses_pts)
            
        if self.with_specific_component('occupancy_head'):
            losses_occupancy = self.occupancy_head.forward_train(results['img_bev_feat'], results=results, gt_occupancy=kwargs['gt_occupancy'], gt_occupancy_flow=gt_occupancy_flow)
            losses.update(losses_occupancy)

        if self.with_specific_component('frpn'):
            losses_mask = self.frpn.get_bev_mask_loss(kwargs['gt_bev_mask'], results['bev_mask_logit'])
            losses.update(losses_mask)

        if self.use_depth_supervision and self.with_specific_component('depth_net'):
            loss_depth = self.depth_net.get_depth_loss(kwargs['gt_depth'], results['depth'])
            losses.update(loss_depth)

        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        self.do_history = True
        if img_inputs is not None:
            for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
                if not isinstance(var, list) :
                    raise TypeError('{} must be a list, but got {}'.format(
                        name, type(var)))        
            num_augs = len(img_inputs)
            if num_augs != len(img_metas):
                raise ValueError(
                    'num of augmentations ({}) != num of image meta ({})'.format(
                        len(img_inputs), len(img_metas)))

            if num_augs==1 and not img_metas[0][0].get('tta_config', dict(dist_tta=False))['dist_tta']:
                return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
            else:
                return self.aug_test(points, img_metas, img_inputs, **kwargs)
        
        elif points is not None:
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        
    def aug_test(self,points,
                    img_metas,
                    img_inputs=None,
                    visible_mask=[None],
                    **kwargs):
        """Test function without augmentaiton."""
        assert False
        return None

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        # outs[0]是检测的结果输出，outs[1]是bev地图的结果输出；
        
        # 获取bev语义map的测评格式
        pred_semantic_indices = self.pts_bbox_head.get_map_indices(outs, img_metas)
        
        return pred_semantic_indices
    
    
    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    visible_mask=[None],
                    return_raw_occ=False,
                    **kwargs):
        """Test function without augmentaiton."""
        results = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        
        if self.backward_projection and self.backward_projection.transformer.encoder.return_intermediate==True:
            pred_map_indices = self.simple_test_pts(results['img_bev_feat_5'], img_metas, rescale=rescale)
        else:
            pred_map_indices = self.simple_test_pts(results['img_bev_feat'], img_metas, rescale=rescale)
        # for result_dict in zip(bbox_list):
        #     print(result_dict)
        #     print(result_dict['pts_bbox'])
        #     result_dict['pts_bbox'] = None
        prediction = {}
        prediction['bbox_results'] = bbox_list
        prediction['pred_map_indices'] = pred_map_indices
        
        return prediction



    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        results = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(results['img_bev_feat'])
        return outs
