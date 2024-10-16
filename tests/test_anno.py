#%%
import os
import open3d as o3d
import numpy as np
import torch
import pickle
import math
from typing import Tuple, List, Dict, Iterable

NOT_OBSERVED = -1
FREE = 0
OCCUPIED = 1

color_map = {
    0: (0,0,0),
    1: (255,255,255),
    2: (255,0,0),
    3: (0,255,0),
    4: (0,0,255),
    5: (255,255,0),
    6: (0,255,255),
    7: (255,0,255),
    8: (192,192,192),
    9: (128,128,128),
    10: (128,0,0),
    11: (128,128,0),
    12: (0,128,0),
    13: (128,0,128),
    14: (0,128,128),
    15: (0,0,128),
 }
colormap_to_colors = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [128, 128, 128, 255], # 16 for vis
], dtype=np.float32)

# def colormap_to_colors(colormap: Dict[str, Iterable[int]]) -> np.ndarray:
#     """
#     Create an array of RGB values from a colormap. Note that the RGB values are normalized
#     between 0 and 1, not 0 and 255.
#     :param colormap: A dictionary containing the mapping from class names to RGB values.
#     :param name2idx: A dictionary containing the mapping form class names to class index.
#     :return: An array of colors.
#     """
#     colors = []
#     for i, (k, v) in enumerate(colormap.items()):
#         # Ensure that the indices from the colormap is same as the class indices.
#         colors.append(v)

#     colors = np.array(colors) / 255  # Normalize RGB values to be between 0 and 1 for each channel.

#     return colors

def voxel2points(voxel, occ_show, voxelSize):
    occIdx = torch.where(occ_show)
    # points = torch.concatenate((np.expand_dims(occIdx[0], axis=1) * voxelSize[0], \
    #                          np.expand_dims(occIdx[1], axis=1) * voxelSize[1], \
    #                          np.expand_dims(occIdx[2], axis=1) * voxelSize[2]), axis=1)
    points = torch.cat((occIdx[0][:, None] * voxelSize[0], \
                        occIdx[1][:, None] * voxelSize[1], \
                        occIdx[2][:, None] * voxelSize[2]), dim=1)
    return points, voxel[occIdx], occIdx

def voxel_profile(voxel, voxel_size):
    centers = torch.cat((voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), dim=1)
    # centers = voxel
    wlh = torch.cat((torch.tensor(voxel_size[0]).repeat(centers.shape[0])[:, None],
                          torch.tensor(voxel_size[1]).repeat(centers.shape[0])[:, None],
                          torch.tensor(voxel_size[2]).repeat(centers.shape[0])[:, None]), dim=1)
    yaw = torch.full_like(centers[:, 0:1], 0)
    return torch.cat((centers, wlh, yaw), dim=1)

def rotz(t):
    """Rotation about the z-axis."""
    c = torch.cos(t)
    s = torch.sin(t)
    return torch.tensor([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def my_compute_box_3d(center, size, heading_angle):
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    heading_angle = -heading_angle - math.pi / 2
    center[:, 2] = center[:, 2] + h / 2
    #R = rotz(1 * heading_angle)
    l, w, h = (l / 2).unsqueeze(1), (w / 2).unsqueeze(1), (h / 2).unsqueeze(1)
    x_corners = torch.cat([-l, l, l, -l, -l, l, l, -l], dim=1)[..., None]
    y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)[..., None]
    z_corners = torch.cat([h, h, h, h, -h, -h, -h, -h], dim=1)[..., None]
    #corners_3d = R @ torch.vstack([x_corners, y_corners, z_corners])
    corners_3d = torch.cat([x_corners, y_corners, z_corners], dim=2)
    corners_3d[..., 0] += center[:, 0:1]
    corners_3d[..., 1] += center[:, 1:2]
    corners_3d[..., 2] += center[:, 2:3]
    return corners_3d

def show_point_cloud(points: np.ndarray, colors=True, points_colors=None, bbox3d=None, voxelize=False, bbox_corners=None, linesets=None, vis=None, offset=[0,0,0]) -> None:
    """
    :param points:
    :param colors: false 不显示点云颜色
    :param points_colors:
    :param bbox3d: voxel边界， Nx7 (center, wlh, yaw=0)
    :param voxelize: false 不显示voxel边界
    :return:
    """
    if vis is None:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
    if isinstance(offset, list) or isinstance(offset, tuple):
        offset = np.array(offset)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    #opt.background_color = np.asarray([0, 0, 0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points+offset)
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(points_colors[:, :3])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.6, origin=[0, 0, 0])

    vis.add_geometry(pcd)
    if voxelize:
        line_sets = o3d.geometry.LineSet()
        line_sets.points = o3d.open3d.utility.Vector3dVector(bbox_corners.reshape((-1, 3))+offset)
        line_sets.lines = o3d.open3d.utility.Vector2iVector(linesets.reshape((-1, 2)))
        line_sets.paint_uniform_color((0, 0, 0))
        # line_sets.colors = o3d.open3d.utility.Vector3dVector(colors)
        # linesets = _draw_bboxes(bbox3d, vis)

    vis.add_geometry(mesh_frame)
    vis.add_geometry(line_sets)
    # vis.run()
    return vis

def main(occ_state, occ_show, voxel_size, vis=None, offset=[0,0,0]):
    # occ_state, voxel_size = data['occ_state'].cpu(), data['voxel_size']
    colors = colormap_to_colors / 255
    pcd, labels, occIdx = voxel2points(occ_state, occ_show, voxel_size)
    _labels = labels % len(colors)
    pcds_colors = colors[_labels]
    bboxes = voxel_profile(pcd, voxel_size)
    bboxes_corners = my_compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])
    #bboxes_corners = torch.cat([my_compute_box_3d(box[0:3], box[3:6], box[6:7])[None, ...] for box in bboxes], dim=0)
    bases_ = torch.arange(0, bboxes_corners.shape[0] * 8, 8)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])  # lines along y-axis
    edges = edges.reshape((1, 12, 2)).repeat(bboxes_corners.shape[0], 1, 1)
    edges = edges + bases_[:, None, None]
    vis = show_point_cloud(points=pcd.numpy(), colors=True, points_colors=pcds_colors, voxelize=True, bbox3d=bboxes.numpy(),
                     bbox_corners=bboxes_corners.numpy(), linesets=edges.numpy(), vis=vis, offset=offset)
    return vis

def generate_the_ego_car():
    ego_range = [-2, -1, 0, 2, 1, 1.5]
    ego_voxel_size=[0.1, 0.1, 0.1]
    ego_xdim = int((ego_range[3] - ego_range[0]) / ego_voxel_size[0])
    ego_ydim = int((ego_range[4] - ego_range[1]) / ego_voxel_size[1])
    ego_zdim = int((ego_range[5] - ego_range[2]) / ego_voxel_size[2])
    ego_voxel_num = ego_xdim * ego_ydim * ego_zdim
    temp_x = np.arange(ego_xdim)
    temp_y = np.arange(ego_ydim)
    temp_z = np.arange(ego_zdim)
    ego_xyz = np.stack(np.meshgrid(temp_y, temp_x, temp_z), axis=-1).reshape(-1, 3)
    ego_point_x = (ego_xyz[:, 0:1] + 0.5) / ego_xdim * (ego_range[3] - ego_range[0]) + ego_range[0]
    ego_point_y = (ego_xyz[:, 1:2] + 0.5) / ego_ydim * (ego_range[4] - ego_range[1]) + ego_range[1]
    ego_point_z = (ego_xyz[:, 2:3] + 0.5) / ego_zdim * (ego_range[5] - ego_range[2]) + ego_range[2]
    ego_point_xyz = np.concatenate((ego_point_y, ego_point_x, ego_point_z), axis=-1)
    ego_points_label =  (np.ones((ego_point_xyz.shape[0]))*16).astype(np.uint8)
    ego_dict = {}
    ego_dict['point'] = ego_point_xyz
    ego_dict['label'] = ego_points_label
    return ego_point_xyz

# %%
path = '/home/chen_jj/workspace/zhujt_workspace/gts/scene-0176/03d958beb9f548399af3d8be08add385/labels.npz'
# %%
data = np.load(path)
# %%
semantic_label = data['semantics']

# %%
import matplotlib.pyplot as plt
plt.imshow(semantic_label[:,:,15])
# %%
# 将0-16映射到不同的plt color
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# 创建一个颜色映射
cmap = cm.get_cmap('viridis', 17)  # 'viridis'是一个颜色映射的名称，17是颜色的数量

# 创建一个0-16的数组
numbers = np.arange(17)

# 将0-16映射到颜色
colors = cmap(numbers)

# 打印颜色
for number, color in zip(numbers, colors):
    print(f"Number: {number}, Color: {color}")
# %%
plt.imshow(semantic_label[:,:,15], cmap=cmap)
# %%
for i in range(16):
    plt.imshow(semantic_label[:,:,i], cmap=cmap)
    plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


# prepare some coordinates
x, y, z = np.indices((8, 8, 8))

# draw cuboids in the top left and bottom right corners, and a link between them
cube1 = (x < 3) & (y < 3) & (z < 3)
cube2 = (x >= 5) & (y >= 5) & (z >= 5)
link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

# combine the objects into a single boolean array
voxels = cube1 | cube2 | link

# set the colors of each object
colors = np.empty(voxels.shape, dtype=object)
colors[link] = 'red'
colors[cube1] = 'blue'
colors[cube2] = 'green'

# and plot everything
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(voxels, facecolors=colors, edgecolor='k')

plt.show()
# %%
import open3d as o3d
import numpy as np

# 创建一个随机的点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))

# 计算体素下采样
voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.05)

# 创建一个体素网格
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(voxel_down_pcd, voxel_size=0.05)

# 绘制原始点云和体素网格
o3d.visualization.draw_geometries([pcd, voxel_grid])
# %%
