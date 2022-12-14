from ._utils import CITY_LABEL_MAP

# GTAV standard training
train_augmentation = dict(
    name='Compose',
    transforms=[
        dict(
            name='ToTensor'
        ),
        dict(
            name='Resize',
            size_image=(1054, 1912),
            size_label=(1054, 1912)
        ),
        dict(
            name='RandomScale',
            min_scale=0.5,
            max_scale=1.5
        ),
        dict(
            name='RandomCrop',
            size=(512, 1024)
        ),
        dict(
            name='RandomHorizontalFlip',
            flip_prob=0.5
        ),
        dict(
            name='Normalize',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        dict(
            name='LabelMap',
            label_id_map=CITY_LABEL_MAP,
            outlier=True
        )
    ]
)
