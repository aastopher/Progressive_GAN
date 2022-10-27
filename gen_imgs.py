import os
from pathlib import Path
import torch.optim as optim
import config
from model import Generator
from utils import (
    load_checkpoint,
    generate_examples,
)

img_size_dict = {0:'4x4', 1:'8x8', 2:'16x16', 3:'32x32', 4:'64x64', 5:'128x128', 6:'256x256', 7:'512x512', 8:'1024x1024'}
img_gen_size = 3
num_samples = 10

gen = Generator(
        config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)

opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))


load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)

if __name__ == "__main__":
    print(f'Generating {num_samples}, {img_size_dict[img_gen_size]} images...')
    generate_examples(gen, img_gen_size, truncation=0.7, n=num_samples)