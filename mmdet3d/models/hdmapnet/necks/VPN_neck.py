import torch
from torch import nn

from ..conmman.homography import bilinear_sampler, IPM
from ..conmman.utils import plane_grid_2d, get_rot_2d, cam_to_pixel
# from .pointpillar import PointPillarEncoder
from ..conmman.base import CamEncode, BevEncode
# from data.utils import gen_dx_bx
from ...builder import NECKS
from mmcv.runner import BaseModule, force_fp32


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx




class ViewTransformation(BaseModule):
    """MLP特征转换

    Args:
        nn (_type_): _description_
    """
    def __init__(self, fv_size, bv_size, n_views=6):
        super(ViewTransformation, self).__init__()
        self.n_views = n_views
        self.hw_mat = []
        self.bv_size = bv_size
        fv_dim = fv_size[0] * fv_size[1]
        bv_dim = bv_size[0] * bv_size[1]
        for i in range(self.n_views):
            fc_transform = nn.Sequential(
                nn.Linear(fv_dim, bv_dim),
                nn.ReLU(),
                nn.Linear(bv_dim, bv_dim),
                nn.ReLU()
            )
            self.hw_mat.append(fc_transform)
        self.hw_mat = nn.ModuleList(self.hw_mat)
        self.con2d = nn.Conv2d(256, 64, kernel_size=1, padding=0)

    def forward(self, feat):
        B, N, C, H, W = feat.shape
        feat = feat.view(B, N, C, H*W)
        outputs = []
        for i in range(N):
            output = self.hw_mat[i](feat[:, i]).view(B, C, self.bv_size[0], self.bv_size[1])
            output = self.con2d(output)
            outputs.append(output)
        outputs = torch.stack(outputs, 1)
        return outputs


@NECKS.register_module()
class VPN_Neck(BaseModule):
    def __init__(self, data_config, grid_config, downsample=16, **kwargs):
        super(VPN_Neck, self).__init__()
        
        fv_size = (data_config['input_size'][0]//downsample, data_config['input_size'][1]//downsample)
        # bv_size = (final_H//2, final_W//2)
        bv_size = (int((grid_config['y'][1] - grid_config['y'][0]) / grid_config['y'][2]) // 4, 
                   int((grid_config['x'][1] - grid_config['x'][0]) / grid_config['x'][2]) // 4)
        self.view_fusion = ViewTransformation(fv_size=fv_size, bv_size=bv_size, n_views=data_config['Ncams'])
   
        self.camC = 64
        self.downsample = downsample
        self.up_sampler = nn.Upsample(size=(200, 200), mode='bilinear', align_corners=True)

        # self.x_bound = [0, data_conf['xbound'][1], data_conf['xbound'][2]]
        
    def forward(self, input):
        x = input[0]
        B,N,C,H,W = x.shape
        
        x = self.view_fusion(x)
        x = x.view(-1,self.camC,x.shape[-2],x.shape[-1])
        x = self.up_sampler(x)
        # x = x.view(B,N,self.camC,200,200)

        return x
