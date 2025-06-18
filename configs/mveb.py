seed: 2048 # random set
launcher: pytorch # pytorch or slurm
port: 28500 # distributed port

# dataset
train_datadir: '/home/user/imagenet/train'
test_datadir: '/home/user/imagenet/val'
n_workers: 5

# model
arch: 'resnet50'
projector_hidden_dim: 2048
projector_output_dim: 2048
num_layers: 3
base_momentum: 0.996
resume_path: 

# optimizer
base_lr: 0.3
whole_batch_size: 4096
momentum: 0.9
weight_decay: 1.0e-6
epochs: 800
warmup_epochs: 10

# loss
loss: 'loss_mveb'
lambd: 0.01  # balance factor for neg grad


# others
print_freq: 50
test_freq: 10
save_freq: 10

# knn config
knn_k: 200
knn_t: 0.1
knn_eval: False