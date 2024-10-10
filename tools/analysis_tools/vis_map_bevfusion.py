#%%
import matplotlib.pyplot as plt
import tqdm
import numpy as np

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap
Root = '/home/chen_jj/workspace/zhujt_workspace/Mono-BEVSeg'

nusc_map = NuScenesMap(dataroot=Root+'/data/nuscenes', map_name='singapore-onenorth')


# %%
# figsize = (12, 4)
# fig, ax = nusc_map.render_map_mask(patch_box, patch_angle, layer_names, canvas_size, figsize=figsize, n_row=1)
# %%
classes = ['drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'divider']
mappings = {}
for name in classes:
    if name == "drivable_area*":
        mappings[name] = ["road_segment", "lane"]
    elif name == "divider":
        mappings[name] = ["road_divider", "lane_divider"]
    else:
        mappings[name] = [name]
# %%
layer_names = []
for name in mappings:
    layer_names.extend(mappings[name])
layer_names = list(set(layer_names))
# %%
patch_box = (300, 1700, 100, 100)
patch_angle = 0  # Default orientation where North is up

canvas_size = (1000, 1000)
map_mask = nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
map_mask[0]
# %%
masks = map_mask.transpose(0, 2, 1)
masks = masks.astype(np.bool)

num_classes = len(classes)
labels = np.zeros((num_classes, *canvas_size), dtype=np.long)
for k, name in enumerate(classes):
    for layer_name in mappings[name]:
        index = layer_names.index(layer_name)
        labels[k, masks[index]] = 1
