# Copyright (c) OpenMMLab. All rights reserved.
# 主要用ipm可视化来debug
#%%
import tempfile

import numpy as np
import torch

from mmdet3d.datasets import NuScenesDataset_with_map
import os

# 获取当前文件的路径
current_file_path = os.path.dirname('/home/chen_jj/workspace/zhujt_workspace/Mono-BEVSeg/')

# 将当前工作目录设置为当前文件的路径
os.chdir(current_file_path)
import torchvision
# %%
# Get the current project directory
import sys
project_dir = '/home/chen_jj/workspace/zhujt_workspace/Mono-BEVSeg'

# Add the project directory to PYTHONPATH
sys.path.append(project_dir)
from mmdet3d.models.hdmapnet.conmman.homography import IPM

#%%
from mmcv import Config
from mmdet3d.datasets import build_dataset, build_dataloader
config_path = '/home/chen_jj/workspace/zhujt_workspace/Mono-BEVSeg/configs/bevdet/geo_split/mono_GW-bevseg-r50_poolv2.py'
cfg = Config.fromfile(config_path)
cfg.data.train['data_root'] = '/home/chen_jj/workspace/zhujt_workspace/Mono-BEVSeg/data/nuscenes/'
cfg.data.train['ann_file'] = '/home/chen_jj/workspace/zhujt_workspace/Mono-BEVSeg/data/nuscenes/bevseg_nusc2_geo_train.pkl'
# cfg.data_config = data_config
dataset = build_dataset(cfg.data.train)

# nus_dataset = NuScenesDataset_with_map(
#     '/home/chen_jj/workspace/zhujt_workspace/Mono-BEVSeg/data/nuscenes/bevseg_nusc_geo_train.pkl',
#     pipeline,
#     '/home/chen_jj/workspace/zhujt_workspace/Mono-BEVSeg/data/nuscenes',)
# data = nus_dataset[0]
#%%
class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
    
    
def get_Ks_RTs_and_post_RTs(cam2imgs, sensor2ego, post_rots, post_trans):
    B, N, _, _ = cam2imgs.shape
    Ks = torch.eye(4, device=cam2imgs.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
    Ks[:,:,:3,:3] = cam2imgs 

    # Rs = torch.eye(4, device=sensor2ego.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
    # Rs[:, :, :3, :3] = sensor2ego[:,:,:3,:3].transpose(-1, -2).contiguous()
    # Ts = torch.eye(4, device=sensor2ego.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
    # Ts[:, :, :3, 3] = -sensor2ego[:,:,:3,3]
    # RTs = Rs @ Ts # 其实就是sensor2ego的逆
    
    # 求sensor2ego的逆
    sensor2ego_inv = torch.eye(4, device=sensor2ego.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
    sensor2ego_inv[:, :, :3, :3] = sensor2ego[:,:,:3,:3]
    sensor2ego_inv[:, :, :3, 3] = sensor2ego[:,:,:3,3]
    sensor2ego_inv = sensor2ego_inv.inverse()
    RTs = sensor2ego_inv
    
    post_RTs = torch.eye(4, device=post_rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
    post_RTs[:, :, :3, :3] = post_rots
    post_RTs[:, :, :3, 3] = post_trans
    
    scale = torch.Tensor([
            [1/4, 0, 0, 0],
            [0, 1/4, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    post_RTs = scale @ post_RTs

    return Ks, RTs, post_RTs

#%%
data = dataset[105]
img = data['img_inputs'][0][0]
import matplotlib.pyplot as plt
denormalize_img = torchvision.transforms.Compose((
    NormalizeInverse(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    torchvision.transforms.ToPILImage(),
))
img = denormalize_img(img)
# img = np.array(img)
plt.imshow(img)
#%%
# 对img进行resize
from torchvision.transforms import Resize
img = Resize((64, 176))(img)
plt.imshow(img)

# %%
inputs = data['img_inputs']

N, C, H, W = inputs[0].shape
_, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs
# imgs.unsqueeze_(0)
#%%
from torchvision.transforms import ToTensor
imgs = ToTensor()(img).unsqueeze(0).unsqueeze(0)
sensor2egos = sensor2egos.view(1, N, 4, 4)
ego2globals = ego2globals.view(1, N, 4, 4)
intrins = intrins.view(1, N, 3, 3)

# calculate the transformation from sweep sensor to key ego
keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)
global2keyego = torch.inverse(keyego2global.double())
sensor2keyegos = \
    global2keyego @ ego2globals.double() @ sensor2egos.double()
sensor2keyegos = sensor2keyegos.float()
# %%
Ks, RTs, post_RTs = get_Ks_RTs_and_post_RTs(intrins, sensor2egos, post_rots, post_trans)




ipm_xbound = [0, 30, 0.15]  # -60 60 0.6 //200
ipm_ybound = [-15, 15, 0.15]  # -30 30 0.6 //100

ipm = IPM(ipm_xbound, ipm_ybound, N=1, C=3, extrinsic=False)

warped_topdown = ipm(imgs.cuda(), Ks.cuda(), RTs.cuda(), None, None, post_RTs=post_RTs.cuda())
warped_topdown_show = warped_topdown[0].permute(1,2,0).cpu().numpy()
# warped_topdown_show = (warped_topdown_show - warped_topdown_show.min()) / (warped_topdown_show.max() - warped_topdown_show.min())

plt.imshow(warped_topdown_show)
plt.show()

# %%
# 读取语义maskgt数据进行可视化

sem_gt_show = data['semantic_map'].permute(1,2,0).cpu().numpy()
semantic_mask = sem_gt_show.astype('uint8') * 255
plt.imshow(semantic_mask)
plt.show()
# %%
# 找出同步的occ标签可视化
plt.imshow(data['gt_occupancy'][:,:,4])
# %%
# plt.hist(data['gt_occupancy'].cpu().numpy().flatten())

# 使用unique函数获取唯一值和它们的数量
unique_values, counts = np.unique(data['gt_occupancy'].cpu().numpy().flatten(), return_counts=True)

# 打印结果
for value, count in zip(unique_values, counts):
    print(f"Value: {value}, Count: {count}")

# %%
