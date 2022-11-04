from importmagician import import_from
with import_from('./'):
    # 1. Import from the corresponding config
    # Or you can just copy-paste (if your config filename includes - or something)
    from configs.lane_detection.bezierlanenet.resnet18_tusimple_style_aug1b import *

    # 2. Define vis_dataset
    from configs.lane_detection.common.datasets._utils import TUSIMPLE_ROOT

vis_dataset = dict(
    name='TuSimpleVis',
    root_dataset=TUSIMPLE_ROOT,
    root_output='./test_tusimple_vis',
    keypoint_json="/root/workspace/code/ENGN8501_LaneDet/output/resnet18_bezierlanenet_tusimple-aug4_mixall_2.json",
    image_set='test'
)
