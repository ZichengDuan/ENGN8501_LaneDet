'''
Author: Yunxiang Liu u7191378@anu.edu.au
Date: 2022-10-14 16:02:43
LastEditors: Yunxiang Liu u7191378@anu.edu.au
LastEditTime: 2022-10-24 02:22:42
FilePath: \pytorch-auto-drive\configs\lane_detection\common\datasets\tusimple_bezier.py
Description: dataset setting
'''
from ._utils import TUSIMPLE_ROOT

dataset = dict(
    name='TuSimpleAsBezier',
    image_set='train',  # Only set for training. Testing will override this value by --state.
    root=TUSIMPLE_ROOT,
    order=3,
    aux_segmentation=True,
    have_aug=True,
    aug_rate=0.2
)
