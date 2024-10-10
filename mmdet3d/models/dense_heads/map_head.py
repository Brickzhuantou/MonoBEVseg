import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.builder import HEADS
from .base_taskhead import BaseTaskHead
import pdb

# from .loss_utils import SegmentationLoss, BinarySegmentationLoss
from mmcv.runner import auto_fp16, force_fp32
from mmdet3d.models.utils import clip_sigmoid
import torchvision
from mmcv.ops.point_sample import bilinear_grid_sample

def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height

    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    bev_resolution = torch.tensor(
        [row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor(
        [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2]
                                 for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension


class BevFeatureSlicer(nn.Module):
    # crop the interested area in BEV feature for semantic map segmentation
    def __init__(self, grid_conf, map_grid_conf):
        super().__init__()

        if grid_conf == map_grid_conf:
            self.identity_mapping = True
        else:
            self.identity_mapping = False

            bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
                grid_conf['x'], grid_conf['y'], grid_conf['z'],
            )

            map_bev_resolution, map_bev_start_position, map_bev_dimension = calculate_birds_eye_view_parameters(
                map_grid_conf['xbound'], map_grid_conf['ybound'], map_grid_conf['zbound'],
            )

            self.map_x = torch.arange(
                map_bev_start_position[0], map_grid_conf['xbound'][1], map_bev_resolution[0])

            self.map_y = torch.arange(
                map_bev_start_position[1], map_grid_conf['ybound'][1], map_bev_resolution[1])

            # convert to normalized coords
            bev_end_position = torch.tensor(
                [row[1] - row[2] / 2.0 for row in [grid_conf['x'], grid_conf['y'], grid_conf['z']]])
            
            self.norm_range_x = bev_end_position[0]
            self.norm_range_y = bev_end_position[1]
            
            # self.norm_map_x = self.map_x / (- bev_start_position[0])
            # self.norm_map_y = self.map_y / (- bev_start_position[1])
            self.norm_map_x = self.map_x / (self.norm_range_x)
            self.norm_map_y = self.map_y / (self.norm_range_y)

            self.map_grid = torch.stack(torch.meshgrid(
                self.norm_map_y, self.norm_map_x), dim=2)

    def forward(self, x):
        # x: bev feature map tensor of shape (b, c, h, w)
        if self.identity_mapping:
            return x
        else:
            grid = self.map_grid.unsqueeze(0).type_as(
                x).repeat(x.shape[0], 1, 1, 1).cuda()

            return F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True) # 这个在onnx中没有对应的算子支持
            # return bilinear_grid_sample(x, grid, align_corners=False) # 这个算子可以支持转onnx


@HEADS.register_module()
class MapHead(BaseTaskHead):
    def __init__(self, task_dict, in_channels, inter_channels=None,
                 train_cfg=None,
                 test_cfg=None,
                 class_weights=[1.0, 2.0, 2.0, 2.0],
                 binary_cls=False,
                 pos_weight=2.0,
                 semantic_thresh=0.25,
                 fcn=False,
                 init_cfg=None,
                 norm_cfg=dict(type='BN'),
                 **kwargs,
                 ):
        super(MapHead, self).__init__(
            task_dict, in_channels, inter_channels, init_cfg, norm_cfg)

        self.semantic_thresh = semantic_thresh
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fcn = fcn

        # loss function for segmentation of semantic map
        self.binary_cls = binary_cls
        if self.binary_cls:
            self.semantic_seg_criterion = BinarySegmentationLoss(
                pos_weight=pos_weight)
        else:
            self.semantic_seg_criterion = SegmentationLoss(
                class_weights=torch.tensor(class_weights).float(),
            )
        if self.fcn:
            trunk = torchvision.models.resnet.resnet18(pretrained=False, zero_init_residual=True)
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
            self.bn1 = trunk.bn1
            self.relu = trunk.relu

            self.layer1 = trunk.layer1
            self.layer2 = trunk.layer2
            self.layer3 = trunk.layer3

            self.up1 = Up(64 + 256, 256, scale_factor=4)
            self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True)
            )

    def get_semantic_indices(self, predictions, targets=None):
        if self.binary_cls:
            pred_semantic_scores = predictions['semantic_seg'].float(
            ).sigmoid()
            pred_semantic_scores, pred_semantic_indices = torch.max(
                pred_semantic_scores, dim=1)
            background_mask = pred_semantic_scores < self.semantic_thresh
            pred_semantic_indices[background_mask] = 0
            pred_semantic_indices = pred_semantic_indices.long()
        else:
            pred_semantic_logits = predictions.clone()
            pred_semantic_indices = torch.argmax(pred_semantic_logits, dim=1)

        return pred_semantic_indices

    @force_fp32(apply_to=('x'))
    def forward(self, x, targets=None):
        x = x[0]
        
        if self.fcn:  # 分割任务加点层?
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            
            x1 = self.layer1(x)
            temp = self.layer2(x1)
            x2 = self.layer3(temp)
            
            x = self.up1(x2, x1)
            x = self.up2(x)
        
        return {task_key: task_head(x) for task_key, task_head in self.task_heads.items()}

    @force_fp32(apply_to=('predictions'))
    def loss(self, predictions, targets):
        loss_dict = {}

        # computing semanatic_map segmentation
        if self.binary_cls:
            assert predictions['semantic_seg'].shape == targets['semantic_map'].shape
            loss_dict['loss_semantic_seg'] = self.semantic_seg_criterion(
                clip_sigmoid(predictions['semantic_seg'].float()),
                targets['semantic_map'].float(),
            )
        else:
            assert predictions['semantic_seg'].shape[-2:
                                                     ] == targets['semantic_seg'].shape[-2:]
            loss_dict['loss_semantic_seg'] = self.semantic_seg_criterion(
                predictions['semantic_seg'].unsqueeze(dim=1).float(),
                targets['semantic_seg'].unsqueeze(dim=1).long(),
            )

        return loss_dict



class BinarySegmentationLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(BinarySegmentationLoss, self).__init__()
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)

        return loss


class SegmentationLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=255, use_top_k=False,
                 top_k_ratio=1.0, future_discount=1.0):

        super().__init__()

        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.future_discount = future_discount

        # self.ce_criterion = nn.CrossEntropyLoss(
        #     weight=self.class_weights, ignore_index=self.ignore_index, reduction='mean')

        # self.nll_criterion = nn.NLLLoss(
        #     weight=self.class_weights, ignore_index=self.ignore_index, reduction='mean')

    def forward(self, prediction, target):
        b, s, c, h, w = prediction.shape
        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w)

        loss = F.cross_entropy(
            prediction,
            target,
            reduction='none',
            weight=self.class_weights.to(target.device).float(),
        )

        # ce_loss = self.ce_criterion(prediction, target)
        # pred_logsoftmax = F.log_softmax(prediction)
        # loss = self.nll_criterion(pred_logsoftmax, target)

        loss = loss.view(b, s, h, w)
        future_discounts = self.future_discount ** torch.arange(
            s, device=loss.device, dtype=loss.dtype)
        future_discounts = future_discounts.view(1, s, 1, 1)
        loss = loss * future_discounts.float()

        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss, _ = torch.sort(loss, dim=2, descending=True)
            loss = loss[:, :, :k]

        return torch.mean(loss)
    
    
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)
