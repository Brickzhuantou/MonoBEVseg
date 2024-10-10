# Copyright (c) Phigent Robotics. All rights reserved.

# mAP: 0.2828
# mATE: 0.7734
# mASE: 0.2884
# mAOE: 0.6976
# mAVE: 0.8637
# mAAE: 0.2908
# NDS: 0.3500
#
# Per-class results:
# Object Class	AP	ATE	ASE	AOE	AVE	AAE
# car	0.517	0.533	0.161	0.123	0.909	0.235
# truck	0.226	0.745	0.232	0.222	0.848	0.268
# bus	0.305	0.797	0.220	0.192	1.982	0.355
# trailer	0.101	1.107	0.230	0.514	0.536	0.068
# construction_vehicle	0.039	1.105	0.501	1.402	0.119	0.386
# pedestrian	0.318	0.805	0.305	1.341	0.826	0.650
# motorcycle	0.216	0.783	0.286	0.977	1.224	0.273
# bicycle	0.203	0.712	0.304	1.354	0.465	0.090
# traffic_cone	0.499	0.547	0.347	nan	nan	nan
# barrier	0.404	0.599	0.297	0.153	nan	nan

_base_ = ['../../_base_/datasets/nus-3d.py', '../../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [0, -15.0, -5.0, 30.0, 15.0, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


data_config = {
    # 先把输入改成前视单目；
    'cams': [
        'CAM_FRONT'
    ],   # !!! 改成多目排除一下多目的问题；
    'Ncams':
    1,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
# 先把检测box和地图的范围保持住，后面pipeline进行过滤处理；
# grid_config = {
#     'x': [0.0, 51.2, 0.8],
#     # 'x': [-51.2, 51.2, 0.8],
    
#     'y': [-51.2, 51.2, 0.8],  # 把分辨率改一下
#     'z': [-5, 3, 8],
#     'depth': [1.0, 60.0, 1.0],
# }

grid_config = {
    'x': [0.0, 30.0, 0.15],
    # 'x': [-51.2, 51.2, 0.8],
    
    'y': [-15.0, 15.0, 0.15],  # 把分辨率改一下
    'z': [-1.0, 5.4, 0.8],
    'depth': [1.0, 60.0, 1.0],
}

map_grid_config_gt = {
    'xbound': [-30.0, 30.0, 0.15],     
    'ybound': [-15.0, 15.0, 0.15],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [1.0, 60.0, 1.0],
}

## config for bevformer
grid_config_bevformer={
    'x': [0, 30, 0.15],
    'y': [-15, 15, 0.15],
    'z': [-1, 5.4, 0.8],
}

occ_grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
}

# map_grid_config_mono = {
#     'xbound': [0.0, 30.0, 0.15],
#     # 'xbound': [-30.0, 30.0, 0.15],      
          
#     'ybound': [-15.0, 15.0, 0.15],
#     'zbound': [-10.0, 10.0, 20.0],
#     'dbound': [1.0, 60.0, 1.0],
# }
map_grid_config_mono = grid_config

voxel_size = [0.1, 0.1, 0.2]

# fb
numC_Trans = 64
use_checkpoint = True
depth_categories = 59
bev_h_ = 200
bev_w_ = 200

# _pos_dim_ = 40
_pos_dim_ = 32

_ffn_dim_ = numC_Trans * 4
_num_levels_= 1

# occ
num_cls = 19
fix_void = num_cls == 19


model = dict(
    type='GWFBSeg',
    readd=True,
    use_depth_supervision=True, # 启用点云深度监督；
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    # img_view_transformer=dict(
    #     type='LSSViewTransformer',
    #     grid_config=grid_config,
    #     input_size=data_config['input_size'],
    #     in_channels=256,
    #     out_channels=numC_Trans,
    #     downsample=16),
    depth_net=dict(
        type='CM_DepthNet', # camera-aware depth net
        in_channels=256,
        context_channels=numC_Trans,
        downsample=16,
        grid_config=grid_config,
        depth_channels=depth_categories,
        with_cp=use_checkpoint,
        loss_depth_weight=1.,
        use_dcn=False,
    ),
    forward_projection=dict(
        type='LSSViewTransformerFunction3D',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        downsample=16),    # 只保留f或者b试试
    # My
    frpn=dict(
        type='FRPN',
        in_channels=numC_Trans,
        scale_factor=1, 
        mask_thre = 0.4,   
        binary_cls=False,
        multi_cls_num = 4,
        class_weights=[1.0, 5.0, 10.0, 5.0],
        semantic_thresh = 0.25,
    ),
    road_occ_head=dict(
        type='RoadOCC',
        in_channels=numC_Trans,
        mask_thre = 0.4,
    ),
    
    # naive
    # frpn = None,
    # road_occ_head = None,
    
    
    backward_projection=dict(
        type='BackwardProjection',
        bev_h=200,
        bev_w=200,
        in_channels=numC_Trans,
        out_channels=numC_Trans,
        pc_range=point_cloud_range,
        transformer=dict(
            type='BEVFormer',
            use_cams_embeds=False,
            embed_dims=numC_Trans,
            num_cams = len(data_config['cams']),
            encoder=dict(
                type='bevformer_encoder',
                num_layers=2,
                pc_range=point_cloud_range,
                grid_config=grid_config_bevformer,
                data_config=data_config,
                return_intermediate=True,
                transformerlayers=dict(
                    type='BEVFormerEncoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=numC_Trans,
                            dropout=0.0,
                            num_levels=1),
                        dict(
                            type='DA_SpatialCrossAttention',
                            num_cams=len(data_config['cams']),
                            pc_range=point_cloud_range,
                            # dbound=[2.0, 42.0, 0.5],
                            dbound=[1.0, 60.0, 1.0],
                            dropout=0.0,
                            deformable_attention=dict(
                                type='DA_MSDeformableAttention',
                                embed_dims=numC_Trans,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=numC_Trans,
                        )
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=numC_Trans,
                        feedforward_channels=_ffn_dim_,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True),),
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
                    # operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
                    # operation_order=('cross_attn', 'norm'))),
           ),
        positional_encoding=dict(
            type='CustormLearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
    ),
    
    
    
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    pts_bbox_head=dict(
        type='Mapseg_Head',
        
        # map分割相关
        cfg_map=dict(
            type='MapHead',
            task_dict={
                'semantic_seg': 4,
            },
            in_channels=256,
            class_weights=[1.0, 5.0, 10.0, 5.0],
            semantic_thresh=0.25,
            fcn=False # 是否选择复杂一点的分割头
        ),
        
        grid_config = grid_config,
        map_grid_config = map_grid_config_mono,
        
        ),
    # model training and testing settings
    
)

# Data
dataset_type = 'NuScenesDataset_with_map'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')
occupancy_path = '/home/jz0424/brick/gts'

# bda_aug_conf = dict(
#     rot_lim=(-22.5, 22.5),
#     scale_lim=(0.95, 1.05),
#     flip_dx_ratio=0.5,
#     flip_dy_ratio=0.5)

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.,
    flip_dy_ratio=0.)   # 取消bev增强看看


train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names, 
        mono=True), 
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='Mono_RasterizeMapVectors',
        # type='RasterizeMapVectors',
        
        map_grid_conf=map_grid_config_gt,
        map_max_channel=3,
        map_thickness=5,
        map_angle_class=36
    ),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='LoadOccupancy', ignore_nonvisible=True, fix_void=fix_void, occupancy_path=occupancy_path, 
         occ_grid_config=occ_grid_config, map_grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'semantic_map','semantic_indices','gt_depth','gt_occupancy'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='Mono_RasterizeMapVectors',
        # type='RasterizeMapVectors',
        
        map_grid_conf=map_grid_config_gt,
        map_max_channel=3,
        map_thickness=5,
        map_angle_class=36
    ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs', 'semantic_map','semantic_indices'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet',
    map_grid_conf=map_grid_config_gt
)

val_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevseg_nusc2_geo_val.pkl') # 为了获取地图标签使用引入了map token的数据；

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevseg_nusc2_geo_test.pkl') # 为了获取地图标签使用引入了map token的数据；

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevseg_nusc2_geo_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=val_data_config,
    test=test_data_config)

for key in ['train', 'val', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=1e-4, 

                 weight_decay=1e-07)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24,])


runner = dict(type='EpochBasedRunner', max_epochs=24)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

evaluation = dict(
    interval=24,
    pipeline=test_pipeline,)

# fp16 = dict(loss_scale='dynamic')

# load_from = '/data1/users/zhujt_workspace/BEVDet-dev2.1/work_dirs/mono_bevdet-r50_poolv2_bevseg/epoch_24_ema.pth'