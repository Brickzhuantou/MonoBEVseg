#%%
# 点云投影处理：
from nuscenes.nuscenes import NuScenes
Root = '/data1/users/zhujt_workspace/BEVDet-dev2.1/'

nusc = NuScenes(version='v1.0-mini', dataroot='/home/jz0424/brick/bevdet/data/nuscenes/nuScenes/', verbose=True)

my_sample = nusc.sample[43]
nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP')
# %%
sensor = 'CAM_FRONT'
cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
cam_front_data
nusc.render_sample_data(cam_front_data['token'])
# %%
