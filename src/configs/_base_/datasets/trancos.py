dataset = dict(
    name='TRANCOS',
    data_root='data/TRANCOS',
    img_norm_cfg=dict(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
    scale=10,
    batch_size=8,
    crop_size=0     # Do not crop
)

