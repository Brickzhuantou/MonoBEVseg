# Copyright (c) Phigent Robotics. All rights reserved.


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
    'z': [-10.0, 10.0, 20.0],
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
    'z': [-1, 5.4, 1.6],
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

numC_Trans = 64


use_checkpoint = True
depth_categories = 60
bev_h_ = 200
bev_w_ = 200

# _pos_dim_ = 40
_pos_dim_ = 32

_ffn_dim_ = numC_Trans * 4
_num_levels_= 1


model = dict(
    type='HDMapNet',
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
    img_view_transformer=dict(
        type='VPN_Neck',
        grid_config=grid_config,
        data_config=data_config,
        downsample=16),
  
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
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'semantic_map','semantic_indices','gt_depth'])
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
    ann_file=data_root + 'bevseg_nusc_geo_val.pkl') # 为了获取地图标签使用引入了map token的数据；

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevseg_nusc_geo_test.pkl') # 为了获取地图标签使用引入了map token的数据；

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevseg_nusc_geo_train.pkl',
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