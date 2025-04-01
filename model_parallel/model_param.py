from myClass.package import *

dataset_path = '../datasets/'
checkpoint_dir = './checkpoints/'
pythonVersion = 'python'
dataset = 'CIFAR10'
num_device = 1

if dataset == "CIFAR10":
    img_size = (32, 32, 3) 
elif dataset == "MNIST": 
    img_size = (32, 32, 1) 
elif dataset == "CELEBA": 
    img_size = (256, 256, 3) 
elif dataset == "CARS": 
    img_size = (128, 128, 3) 

batch_size = 8
inference_batch_size = 8
epochs = 1000
patience = 300
lr = 2e-4
save_image_epoch = 10

model_config = {
    'im_channels': img_size[2],                     # Number of input channels
    'down_channels': [16, 16, 32, 32, 64],          # Number of channels in Downsample blocks
    'mid_channels': [64, 64, 32],                   # Number of channels in Mid blocks
    'time_emb_dim': 128,                            # Embedding dimension
    'down_sample': [True, True, True, True],        # Choose in which downsample block downsample the images
    'num_down_layers': 2,                           # Number of levels in each Downsample block
    'num_mid_layers': 2,                            # Number of levels in each Mid block
    'num_up_layers': 2,                             # Number of levels in each Upsample block
    'dropout': 0,                                   # Dropout rate

    # Choose where to use swin block 
    'apply_attention_down': [False, False, False, True,],    # Apply Cross-Attention in Downsample block
    'apply_attention_mid': [True, True],                     # Apply Cross-Attention in Mid block
    'apply_attention_up': [False, False, False, True],       # Apply Cross-Attention in Upsample block
}