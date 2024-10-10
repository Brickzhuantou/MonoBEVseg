import torch
from torch import nn

from mmcv.runner import BaseModule, force_fp32

from ..conmman.homography import bilinear_sampler, IPM
from ...builder import NECKS



@NECKS.register_module()
class IPM_Neck(BaseModule):
    def __init__(self, data_config, grid_config, downsample=16, **kwargs):
        super(IPM_Neck, self).__init__()
        
        self.con1x1 = nn.Conv2d(256, 64, kernel_size=1, padding=0)
        
        self.ipm = IPM(grid_config['x'], grid_config['y'], N=1, C=64, extrinsic=False)
 
        self.camC = 64
        self.downsample = downsample

        # self.x_bound = [0, data_conf['xbound'][1], data_conf['xbound'][2]]

        # dx, bx, nx = gen_dx_bx(self.
    
    def get_Ks_RTs_and_post_RTs(self, cam2imgs, sensor2ego, post_rots, post_trans):
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
                [1/self.downsample, 0, 0, 0],
                [0, 1/self.downsample, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]).cuda()
        post_RTs = scale @ post_RTs

        return Ks, RTs, post_RTs
        
    def forward(self, input):
        x = input[0]
        sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda = input[1:]
        B,N,C,H,W = x.shape
        x = x.view(B*N,C,H,W)
        x = self.con1x1(x)
        x = x.view(B,N,self.camC,H,W)
        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(cam2imgs, sensor2ego, post_rots, post_trans)
        topdown = self.ipm(x, Ks, RTs, None, None, post_RTs)
        return topdown
