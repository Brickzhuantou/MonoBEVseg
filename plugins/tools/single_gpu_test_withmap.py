# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp

import mmcv
import torch
from mmcv.image import tensor2imgs

from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)
from ..metrics import IntersectionOverUnion, PanopticMetric

def single_gpu_test_withmap(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool, optional): Whether to save viualization results.
            Default: True.
        out_dir (str, optional): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    det_results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    # map评测指标
    num_map_class = 4
    semantic_map_iou_val = IntersectionOverUnion(num_map_class).cuda()
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(
                    data,
                    result,
                    out_dir=out_dir,
                    show=show,
                    score_thr=show_score_thr)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
        # 判断result['bbox_results'][0]是不是空字典
        if result['bbox_results'][0]:
            result['bbox_results'][0]['pts_bbox']['pred_map_indices'] = result['pred_map_indices']
            result['bbox_results'][0]['pts_bbox']['gt_map_indices'] = data['semantic_indices'][0]   # 把map分割结果和gt也送到最后的
        else:
            result['bbox_results'][0] = {'pts_bbox': {'pred_map_indices': result['pred_map_indices'], 'gt_map_indices': data['semantic_indices'][0]}}
        det_results.extend(result['bbox_results'])
        
        
        # 计算map的IoU指标=============================
        pred_semantic_indices = result['pred_map_indices']
        target_semantic_indices = data['semantic_indices'][0].cuda()  # 这个0的用处待考察
        semantic_map_iou_val(pred_semantic_indices,
                                target_semantic_indices)
        
        # =============================================

        batch_size = len(result['bbox_results'])
        for _ in range(batch_size):
            prog_bar.update()
            
    scores = semantic_map_iou_val.compute()
    mIoU = sum(scores[1:]) / (len(scores) - 1)
    print('[Validation {:04d} / {:04d}]: semantic map iou = {}, mIoU = {:.3f}'.format(
        len(dataset), len(dataset), scores, mIoU,
    ))
    return det_results  # 暂时只返回检测结果
