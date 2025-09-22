# RTMDet-tiny Configuration for Augmented Package Dataset
# Optimized for training on generated augmented dataset

_base_ = ['rtmdet_tiny_8xb32-300e_coco.py']

# Model configuration for package detection
model = dict(
    bbox_head=dict(num_classes=1)  # Package class only
)

# Dataset configuration for augmented data
dataset_type = 'CocoDataset'
data_root = 'augmented_data_production/'

train_dataloader = dict(
    batch_size=16,
    num_workers=16,  # Optimized for high-core systems
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/images/')
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='valid/annotations.json', 
        data_prefix=dict(img='valid/images/')
    )
)

# Optimizer for augmented dataset training
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05)
)

# Training configuration
train_cfg = dict(max_epochs=100, val_interval=10)
work_dir = 'work_dirs/rtmdet_tiny_augmented_package_detection'
