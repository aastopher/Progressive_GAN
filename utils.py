from email.policy import default
import torch
import random
import numpy as np
import os
from pathlib import Path
import torchvision
import torch.nn as nn
import config
import torch.optim as optim
from model import Generator
from torchvision.utils import save_image
from scipy.stats import truncnorm
import click
import gdown
import zipfile
import imagehash
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from math import log2
from tqdm import tqdm
import pandas as pd

# def get_loader(image_size):
#     transform = transforms.Compose(
#         [
#             transforms.Resize((image_size, image_size)),
#             transforms.ToTensor(),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.Normalize(
#                 [0.5 for _ in range(config.CHANNELS_IMG)],
#                 [0.5 for _ in range(config.CHANNELS_IMG)],
#             ),
#         ]
#     )

#     batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
#     dataset = datasets.ImageFolder(root=config.DATASET, transform=transform)
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=config.NUM_WORKERS,
#         pin_memory=True,
#     )
    
#     return loader, dataset

def get_loader(image_size):
    transform = transforms.Compose(
        [
            # transforms.functional.resize(size=1024),
            transforms.Resize(1024),
            # transforms.RandomCrop((1024, 1024)),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)],
                [0.5 for _ in range(config.CHANNELS_IMG)],
            ),
        ]
    )

    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=config.DATASET, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    
    return loader, dataset

# Print losses occasionally and print to tensorboard
def plot_to_tensorboard(writer, loss_critic, loss_gen, real, fake, img_size, tensorboard_step):
    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)
    writer.add_scalar("Loss Generator", loss_gen, global_step=tensorboard_step)
    writer.add_scalar("Image Size", img_size, global_step=tensorboard_step)

    with torch.no_grad():
        # take out (up to) 8 examples to plot
        img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
        writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
        writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)


def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    # checkpoint = torch.load(checkpoint_file, map_location="cuda")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#### CLI Functions ####
def remove_dups():
    '''Inspired from https://github.com/JohannesBuchner/imagehash repository'''

    def hash_img(hashfunc, path, hash_size=8):
            '''takes an image path, returns a hash value'''

            image = Image.open(path)
            # print(path)

            # remove alpha
            if image.mode != 'RGBA':
                pass
            else:
                canvas = Image.new('RGBA', image.size, (255,255,255,255))
                canvas.paste(image, mask=image)
                image = canvas.convert('RGB')

            # collect zdata
            image = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
            data = image.getdata()
            quantiles = np.arange(100)
            quantiles_values = np.percentile(data, quantiles)
            zdata = (np.interp(data, quantiles_values, quantiles) / 100 * 255).astype(np.uint8)
            image.putdata(zdata)

            return hashfunc(image)

    # collect all image_name, hash pairs
    img_list = os.listdir(Path(f"{config.DATASET}/imgs/"))
    img_hash_list = []
    for image_name in img_list:
        img_path = Path(f"{config.DATASET}/imgs/{image_name}")
        img_hash = hash_img(imagehash.dhash, img_path, hash_size = 8) # adjust hash_size to change duplicate distance
        img_hash_list.append((image_name, str(img_hash)))

    # collect duplicate lists and remove second duplicate files
    img_hash_df = pd.DataFrame(img_hash_list, columns=['image_name', 'img_hash'])
    first_dup_list = img_hash_df[img_hash_df['img_hash'].duplicated(keep = 'first')]['image_name'].to_list()
    second_dup_list = img_hash_df[img_hash_df['img_hash'].duplicated(keep = 'last')]['image_name'].to_list()
    print(f'removing {len(second_dup_list)} duplicate images...')
    for dup in second_dup_list:
        os.remove(Path(f"{config.DATASET}/imgs/{dup}"))

    # show duplicate image paths
    for first, second in list(zip(first_dup_list, second_dup_list)):
        print(Path(f"{config.DATASET}/imgs/{first}"))
        print(Path(f"{config.DATASET}/imgs/{second}"))
    

def download_data():
    # training imgs download url + output file name definition
    url = 'https://drive.google.com/uc?id=1Jn-FOKZ6LoRkhXP3jwag_PupceV-KEvY'
    outfile = "imgs.zip"

    # download imgs if imgs folder does not exist
    if not os.path.exists(config.DATASET):
        gdown.download(url, outfile, quiet=False)

        with zipfile.ZipFile(outfile, 'r') as zip_ref:
            zip_ref.extractall()
        os.remove("imgs.zip")
        os.rename("cybercity_imgs","imgs")

def generate_samples(args):
    """
    Tried using truncation trick here but not sure it actually helped anything, you can
    remove it if you like and just sample from torch.randn
    """
    num_samples, img_size = args
    img_size_dict = {0:'4x4', 1:'8x8', 2:'16x16', 3:'32x32', 4:'64x64', 5:'128x128', 6:'256x256', 7:'512x512', 8:'1024x1024'}
    gen = Generator(
            config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
        ).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
    print(f'Generating {num_samples}, {img_size_dict[img_size]} images...')

    gen.eval()
    alpha = 1.0
    truncation = 0.7
    for i in range(num_samples):
        with torch.no_grad():
            noise = torch.tensor(truncnorm.rvs(-truncation, truncation, size=(1, config.Z_DIM, 1, 1)), device=config.DEVICE, dtype=torch.float32)
            # noise = torch.randn(1, config.Z_DIM, 1, 1).to(config.DEVICE)
            img = gen(noise, alpha, img_size)
            if not os.path.exists(config.RESULTS): # check if results folder exists
                os.makedirs(config.RESULTS) # create results folder if does not exist
            print(Path(f"{config.RESULTS}/img_{i}.png"))
            save_image(img*0.5+0.5, Path(f"{config.RESULTS}/img_{i}.png"))
    gen.train()

def prev_imgs():
    '''yields first real image batch to results folder'''
    loader, _ = get_loader(1024)
    data, _ = next(iter(loader))
    for i in range(data.size(0)):
           torchvision.utils.save_image(data[i, :, :, :]*0.5+0.5, Path(f"{config.RESULTS}/img_{i}.png"))

# CLI Driver ##
@click.command()
@click.argument('option', required=False)
@click.argument('args', required=False, nargs=-1)
def cli(args, option):
    commands = ['sample', 'download', 'removedups','transform']

    if not option:
        print(f'no option provided')
    elif option not in commands:
        print(f'invalid option: {option}')
    elif option == 'sample':
        if not args:
            args = (10,4)
        else:
            args = tuple(map(int, args))
        print(f'sample: {args}')
        generate_samples(args)
    elif option == 'download':
        print(f'option: {option}')
        download_data()
    elif option == 'removedups':
        print(f'option: {option}')
        remove_dups()
    elif option == 'transform':
        print(f'option: {option}')
        prev_imgs()

if __name__ == "__main__":
    cli()