from ._utils import CULANE_ROOT

dataset = dict(
    name='CULaneAsBezier_style',
    image_set='train',  # Only set for training. Testing will override this value by --state.
    root=CULANE_ROOT,
    order=3,
    aux_segmentation=True,
    style='winter'
)
