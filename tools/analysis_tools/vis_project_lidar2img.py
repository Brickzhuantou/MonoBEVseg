# %%
# 点云投影处理：
from nuscenes.nuscenes import NuScenes
Root = '/data1/users/zhujt_workspace/BEVDet-dev2.1/'

nusc = NuScenes(version='v1.0-mini', dataroot='/data2/sxkc/nuscenes/trainval/', verbose=True)

my_sample = nusc.sample[43]
nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP')
# %%
sample_token = my_sample['token']
sample_record = nusc.get('sample', sample_token)
pointsensor_token = sample_record['data']['LIDAR_TOP']
camera_token = sample_record['data']['CAM_FRONT']

# %%
import os.path as osp
cam = nusc.get('sample_data', camera_token)
pointsensor = nusc.get('sample_data', pointsensor_token)
pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
# %%
# 读取点云数据
import numpy as np
scan = np.fromfile(pcl_path, dtype=np.float32)
points = scan.reshape((-1,5))[:,:4]
points = points.T
# %%
# 读取图像数据
from PIL import Image
im = Image.open(osp.join(nusc.dataroot, cam['filename']))
# %%
# 点云转到自车坐标系(点云时间戳)
from pyquaternion import Quaternion

cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
points[:3, :] = np.dot(Quaternion(cs_record['rotation']).rotation_matrix, points[:3,:])
for i in range(3):
    points[i, :] = points[i, :] + np.array(cs_record['translation'])[i]

# #%% 想看看时间戳不一致对结果影响大不大
# # 点云从自车转到全局坐标系
# poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
# points[:3, :] = np.dot(Quaternion(poserecord['rotation']).rotation_matrix, points[:3,:])
# for i in range(3):
#     points[i, :] = points[i, :] + np.array(poserecord['translation'])[i]
# #%%
# # 全局坐标系转回到自车坐标系（相机时间戳）
# poserecord = nusc.get('ego_pose', cam['ego_pose_token'])

# for i in range(3):
#     points[i, :] = points[i, :] - np.array(poserecord['translation'])[i]
# points[:3, :] = np.dot(Quaternion(poserecord['rotation']).rotation_matrix.T, points[:3,:])


# %%
# 将点云从自车坐标系转到相机坐标系
cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
for i in range(3):
    points[i, :] = points[i, :] - np.array(cs_record['translation'])[i]
points[:3, :] = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T, points[:3,:])

# %%
# 转到像素坐标系画出来

def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points
depths = points[2, :]

points = view_points(points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

coloring = depths
min_dist = 1.0
mask = np.ones(depths.shape[0], dtype=bool)
mask = np.logical_and(mask, depths > min_dist)
mask = np.logical_and(mask, points[0, :] > 1)
mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
mask = np.logical_and(mask, points[1, :] > 1)
mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
points = points[:, mask]
coloring = coloring[mask]

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(9, 16))
# fig.canvas.set_window_title(sample_token)
ax.imshow(im)
ax.scatter(points[0, :], points[1, :], c=coloring, s=5)
ax.axis('off')
plt.show()

# %%
