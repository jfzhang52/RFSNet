optimizer = dict(
    type='Adam',
    lr=1e-4,
    weight_decay=0
)
runner = dict(
    max_epochs=500,
    start_epoch=0,
    print_freq=10,
    val_freq=1,
    num_workers=8,
    device=0,
    resume='',
    base_dir='',
    ckpt_dir=''
)
