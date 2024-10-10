# 对map标签进行可视化；

#%%
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
Root = '/home/jz0424/brick/bevdet'
nusc_map = NuScenesMap(dataroot=Root+'/data/nuscenes', map_name='singapore-onenorth')
nusc = NuScenes(version='v1.0-trainval', dataroot=Root+'/data/nuscenes', verbose=True)

#%%
my_sample = nusc.sample[130]
sample_token = my_sample['token']
sample_record = nusc.get('sample', sample_token)
# pointsensor_token = sample_record['data']['LIDAR_TOP']
camera_token = sample_record['data']['CAM_FRONT']

sd_rec = nusc.get('sample_data', camera_token)
map_location = nusc.get('log', nusc.get('scene', sample_record['scene_token'])['log_token'])['location']
pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
ego2global_translation = pose_record['translation']
ego2global_rotation = pose_record['rotation']

#%%
# 引入map相关参数
from mmdet3d.datasets.map_utils import VectorizedLocalMap, preprocess_map

map_grid_config = {
    'xbound': [-30.0, 30.0, 0.15],
    'ybound': [-15.0, 15.0, 0.15],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [1.0, 60.0, 1.0],
}
map_data_root= '/home/jz0424/brick/bevdet/data/nuscenes/'
map_xbound, map_ybound = map_grid_config['xbound'], map_grid_config['ybound']
patch_h = map_ybound[1] - map_ybound[0]
patch_w = map_xbound[1] - map_xbound[0]
canvas_h = int(patch_h / map_ybound[2])
canvas_w = int(patch_w / map_xbound[2])
map_patch_size = (patch_h, patch_w)
map_canvas_size = (canvas_h, canvas_w)
# %%
import numpy as np
vector_map = VectorizedLocalMap(
            dataroot=map_data_root,
            patch_size=map_patch_size,
            canvas_size=map_canvas_size,
        )

vectors = vector_map.gen_vectorized_samples(
        map_location, ego2global_translation, ego2global_rotation)
# 这里得到的矢量点都是基于自车坐标系为原点的范围内选取的；

for vector in vectors:
    pts = vector['pts']
    vector['pts'] = np.concatenate(
        (pts, np.zeros((pts.shape[0], 1))), axis=1)
    
    
#%%
# 过滤像素平面外的点；
#%%
import torch
from pyquaternion import Quaternion
# 读取相机内外参
sens = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
trans=(torch.Tensor(sens['translation']))
rots=(torch.Tensor(Quaternion(sens['rotation']).rotation_matrix))
intrins=(torch.Tensor(sens['camera_intrinsic']))
fu, cu = intrins[0, 0].numpy(), intrins[0, 2].numpy()
#%%
# 先将要素点集从自车坐标系转到相机坐标系，再将点投影到前视图进行mask；

# 根据可视区域过滤；

new_vectors = []
for vector in vectors:
    pts = vector['pts']  # 012分别表示前后（前为正），左右（左为正），高度；
    
    # 把自车坐标系的点转到相机坐标系
    for i in range(3):
        pts[:, i] = pts[:, i] - np.array(trans)[i]
    pts = np.dot(rots.T, pts.transpose(1,0)[:3,:]).transpose(1,0)
    
    # 然后012表示左右（右为正），上下，前后（前为正）
    uc = pts[:,0]/pts[:,2] *fu +cu
    
    index = (pts[:,2]>0) & (uc<1600) & (uc>0) 
    vector['pts'] = pts[index==True] # 要保留index下的点
    
    # 将vector的坐标再转回相机坐标系用于后续可视化；
    vector['pts'] = np.dot(rots, vector['pts'].transpose(1,0)[:3,:]).transpose(1,0)
    for i in range(3):
        vector['pts'][:, i] = vector['pts'][:, i] + np.array(trans)[i]
    
    if vector['pts'].shape[0] > 2:
        new_vectors.append(vector)
    



#%%
# 获取语义栅格图
map_max_channel = 3
map_thickness = 5
map_angle_class=36

vectors = new_vectors
for vector in vectors:
    vector['pts'] = vector['pts'][:, :2]

semantic_masks, instance_masks, forward_masks, backward_masks = preprocess_map(
            vectors, map_patch_size, map_canvas_size, map_max_channel, map_thickness, map_angle_class)
num_cls = semantic_masks.shape[0]
indices = np.arange(1, num_cls + 1).reshape(-1, 1, 1)
semantic_indices = np.sum(semantic_masks * indices, axis=0)

#%%
import matplotlib.pyplot as plt
semantic_mask = semantic_masks.astype('uint8') * 255
semantic_mask = np.moveaxis(semantic_mask, [0,1], [-1,-2])
# plt.figure(figsize=(2,4))
plt.xlim(0, 200)
plt.ylim(0, 400)
plt.imshow(semantic_mask)
# %%
import os.path as osp
from PIL import Image
cam = nusc.get('sample_data', camera_token)

im = Image.open(osp.join(nusc.dataroot, cam['filename']))
im
# %%
