import time
from typing import Dict, Optional, Sequence, Union

import tensorrt as trt
import torch
import torch.onnx
from mmcv import Config
from mmdeploy.backend.tensorrt import load_tensorrt_plugin

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

import argparse
from mmcv import Config, DictAction
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from plugins.metrics import IntersectionOverUnion, PanopticMetric


def parse_args():
    parser = argparse.ArgumentParser(description='Deploy BEVDet with Tensorrt')
    parser.add_argument('config', help='deploy config file path')
    parser.add_argument('engine', help='checkpoint file')
    parser.add_argument('--samples', default=80, help='samples to benchmark')
    parser.add_argument('--postprocessing', action='store_true')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')

    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
             ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')

    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function')
    args = parser.parse_args()
    return args


def torch_dtype_from_trt(dtype: trt.DataType) -> torch.dtype:
    """Convert pytorch dtype to TensorRT dtype.

    Args:
        dtype (str.DataType): The data type in tensorrt.

    Returns:
        torch.dtype: The corresponding data type in torch.
    """

    if dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError(f'{dtype} is not supported by torch')


class TRTWrapper(torch.nn.Module):

    def __init__(self,
                 engine: Union[str, trt.ICudaEngine],
                 output_names: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.engine = engine
        if isinstance(self.engine, str):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                with open(self.engine, mode='rb') as f:
                    engine_bytes = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        names = [_ for _ in self.engine]
        input_names = list(filter(self.engine.binding_is_input, names))
        self._input_names = input_names
        self._output_names = output_names

        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    def forward(self, inputs: Dict[str, torch.Tensor]):
        bindings = [None] * (len(self._input_names) + len(self._output_names))
        for input_name, input_tensor in inputs.items():
            idx = self.engine.get_binding_index(input_name)
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

            # create output tensors
        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))

            device = torch.device('cuda')
            output = torch.zeros(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()
        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)
        return outputs


def get_plugin_names():
    return [pc.name for pc in trt.get_plugin_registry().plugin_creator_list]


def get_semantic_indices(predictions):
    
    pred_semantic_logits = predictions.clone()
    pred_semantic_indices = torch.argmax(pred_semantic_logits, dim=1)

    return pred_semantic_indices



def main():

    load_tensorrt_plugin()

    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + 'TRT'
    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [0]

    # build dataloader
    assert cfg.data.test.test_mode
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    # build tensorrt model
    # trt_model = TRTWrapper(args.engine, [f'output_{i}' for i in range(36)])
    trt_model = TRTWrapper(args.engine,
                           [f'output_{i}' for i in
                            range(7 * len(model.pts_bbox_head.task_heads))])
    num_warmup = 5
    pure_inf_time = 0

    init_ = True
    metas = dict()
    outputs = []
    num_map_class = 4
    semantic_map_iou_val = IntersectionOverUnion(num_map_class).cuda()
    
    # benchmark with several samples and take the average
    for i, data in enumerate(data_loader):
        if init_:
            inputs = [t.cuda() for t in data['img_inputs'][0]]
            metas_ = model.get_bev_pool_input(inputs)
            metas = dict(
                ranks_bev=metas_[0].int().contiguous(),
                ranks_depth=metas_[1].int().contiguous(),
                ranks_feat=metas_[2].int().contiguous(),
                interval_starts=metas_[3].int().contiguous(),
                interval_lengths=metas_[4].int().contiguous())
            init_ = False
        img = data['img_inputs'][0][0].cuda().squeeze(0).contiguous()
        
        
        torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        trt_output = trt_model.forward(dict(img=img, **metas))

        # postprocessing
        if args.postprocessing:
            # trt_output = [trt_output[f'output_{i}'] for i in range(36)]
            trt_output = [trt_output[f'output_{i}'] for i in
                          range(7 * len(model.pts_bbox_head.task_heads))]
            det_pred, map_pred = model.result_deserialize(trt_output)
            img_metas = [dict(box_type_3d=LiDARInstance3DBoxes)]
            bbox_list = model.pts_bbox_head.get_bboxes(
                det_pred, img_metas, rescale=True)
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
            
           
            prediction = {}
            
            result_dict = {}
            result_dict['pts_bbox'] = bbox_results[0]
            # outputs.append(result_dict)
            
            prediction['bbox_results'] = [result_dict]
            
            # 读取语义标签：
            pred_semantic_indices = get_semantic_indices(map_pred)
            prediction['pred_map_indices'] = pred_semantic_indices
            target_semantic_indices = data['semantic_indices'][0].cuda() 
            semantic_map_iou_val(pred_semantic_indices,
                                target_semantic_indices)
            
            prediction['bbox_results'][0]['pts_bbox']['pred_map_indices'] = pred_semantic_indices
            prediction['bbox_results'][0]['pts_bbox']['gt_map_indices'] = data['semantic_indices'][0]
            
            outputs.extend(prediction['bbox_results'])
            
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % 5 == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done image [{i + 1:<3}/ {args.samples}], '
                      f'fps: {fps:.1f} img / s')

        if (i + 1) == args.samples:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall \nfps: {fps:.1f} img / s '
                  f'\ninference time: {1000/fps:.1f} ms')
            # return fps
    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        dataset.format_results(outputs, **kwargs)
    if args.eval:
        
        
        scores = semantic_map_iou_val.compute()
        mIoU = sum(scores[1:]) / (len(scores) - 1)
        print('[Validation {:04d} / {:04d}]: semantic map iou = {}, mIoU = {:.3f}'.format(
            len(dataset), len(dataset), scores, mIoU,
        ))
        
        eval_kwargs = cfg.get('evaluation', {}).copy()
        # hard-code way to remove EvalHook args
        for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule'
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        print(dataset.evaluate(outputs, **eval_kwargs))

if __name__ == '__main__':
    fps = main()
