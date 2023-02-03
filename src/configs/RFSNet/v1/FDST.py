_base_ = [
    '../../_base_/models/RFSNet.py',
    '../../_base_/datasets/fdst.py',
    '../../_base_/schedules/schedule_500.py',
    '../../_base_/default_runtime.py'
]

model = dict(
    name='RFSNet_v1',
    type='V',
    num_blocks=4,
    use_bn=False,
    lw=True,
)

dataset = dict(
    name='FDST',
    data_root='data/FDST',
    batch_size=1,
    scale=5,    # time step
)

optimizer = dict(
    type='Adam',
    lr=1e-4,
)

runner = dict(
    print_freq=400,
    max_epochs=100,
    num_workers=4,
)
