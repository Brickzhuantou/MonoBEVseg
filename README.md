# Mono_BEVSeg

## 1. 环境配置
参考BEVDet中的环境配置部分即可；
## 2. 数据预处理（pickle文件生成）
1. 先执行python ./tools/create_data_bevdet.py生成原始的检测pickle文件；
2. 执行python ./tools/add_maptoken_to_bevdet_data.py对pickle进行补充（添加了map标签制作所需要的位姿token）

## 3. 运行指令
### 训练指令示例：
  CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh
./configs/bevdet/mono_bevseg-r50_poolv2.py 2 --
work-dir=./work_dirs
### 测试指令
python tools/test.py ./configs/bevdet/mono_bevseg-r50_poolv2.py ./work_dirs/LSS_base/epoch_24_ema.pth --eval mAP --seg_eval --eval-options jsonfile_prefix='json文件的存储路径'

包含--eval-options选项可以将json文件保存下来；

### 可视化
python tools/analysis_tools/vis.py [测试阶段生成的results_nusc.json的存储路径] --save_path[视频存储路径]

# TODO:
1. 测试验证；
2. config整理；


