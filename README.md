# Dall_e_tutorial
Purpose of this repository is to practice implementation of Dall-E

> Simply installing Dall-E Package is a way to apply and use Dall-E. However, in this tutorial I would like to look into original codes, to study how Dall-E package is designed. Also by doing this, I would like to try modifying some codes to enable training my own model for my own data. 

[1. Original code & Reference](#Original-code-&-Reference).  
[2. Setup in local environment](#Setup-in-local-environment).  

## Original code & Reference
- https://github.com/lucidrains/DALLE-pytorch

## Setup in local environment
According to the [original source](https://github.com/lucidrains/DALLE-pytorch), training Dall-E model two essential steps:   
1. Traind VAE using DiscreteVAE in dalle_pytorch module

'''
import torch
from dalle_pytorch import DiscreteVAE

vae = DiscreteVAE(
    image_size = 256,
    num_layers = 3,           # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
    num_tokens = 8192,        # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
    codebook_dim = 512,       # codebook dimension
    hidden_dim = 64,          # hidden dimension
    num_resnet_blocks = 1,    # number of resnet blocks
    temperature = 0.9,        # gumbel softmax temperature, the lower this is, the harder the discretization
    straight_through = False, # straight-through for gumbel softmax. unclear if it is better one way or the other
)

images = torch.randn(4, 3, 256, 256)

loss = vae(images, return_loss = True)
loss.backward()

# train with a lot of data to learn a good codebook
'''

2. 
