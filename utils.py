from cmath import inf
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
from torch.utils.data import DataLoader, SubsetRandomSampler
from math import log2
from tqdm import tqdm
import pandas as pd

def get_loader(image_size):
    # transform = transforms.Compose(
    #     [
    #         # transforms.Resize(512),
    #         # transforms.RandomCrop((512, 512)),
    #         transforms.Resize((image_size, image_size)),
    #         transforms.ToTensor(),
    #         transforms.RandomHorizontalFlip(p=0.5),
    #         transforms.Normalize(
    #             [0.5 for _ in range(config.CHANNELS_IMG)],
    #             [0.5 for _ in range(config.CHANNELS_IMG)],
    #         ),
    #     ]
    # )

    transform = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.RandomCrop((512, 512)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=(0.1, 0.4), hue=(-0.5, 0.5)),
            transforms.RandomAffine(degrees=(0, 60), scale=(0.75, 1)),
            transforms.CenterCrop(256),
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
    
    # init list in range of the min between config.SAMPLE_SIZE and len(dataset)
    dataset_indices = list(range(min(config.SAMPLE_SIZE, len(dataset))))
    np.random.shuffle(dataset_indices) # shuffle indices 
    sample = SubsetRandomSampler(dataset_indices) # init sampler

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        sampler=sample,
    )
    
    return loader, dataset

# Print losses occasionally and print to tensorboard
def plot_to_tensorboard(writer, loss_critic, loss_gen, fid, real, fake, img_size, tensorboard_step):
    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)
    writer.add_scalar("Loss Generator", loss_gen, global_step=tensorboard_step)
    writer.add_scalar("FID", fid, global_step=tensorboard_step)
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
def init():
    '''download logs for exploration then init empty models and results directory'''
    log_url = 'https://drive.google.com/uc?id=1-GbQp4cXmqVslMxBbirJEwf2SgWEDZlb'
    outfile = "logs.zip"

    # download logs if logs folder does not exist
    if not os.path.exists(config.LOGS_PATH):
        gdown.download(log_url, outfile, quiet=False)

        with zipfile.ZipFile(outfile, 'r') as zip_ref:
            zip_ref.extractall()

        os.rename("ProGAN_logs","logs")
        os.mkdir(config.MODEL_PATH)
        os.mkdir(config.RESULTS)
        os.remove("logs.zip")

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
    

def download_models(args):
    '''download pre-trained models and images for various models''' 

    if args == 'cars':
        models_url = 'https://drive.google.com/uc?id=1-2pczU0Vsx61ru6aYJuwaV-By4Mdqarj' # ProGAN_Cars.zip
        imgs_url = 'https://drive.google.com/uc?id=1l0liZMZV3PGDonJS8FcNq_-5W9oiVP9B' # car_imgs.zip
        img_name = 'car_imgs'
        model_name = 'ProGAN_Cars'
    elif args == 'cyber':
        models_url = 'https://drive.google.com/uc?id=1-C5Up7hPiB9F0KJ1FeqteKMnkhHZGa1Z' # ProGAN_Cyber.zip
        imgs_url = 'https://drive.google.com/uc?id=1-BPlYeT0WKXeM1I8NuGSqL84iwZdAuf2' # cybercity_imgs.zip
        img_name = 'cybercity_imgs'
        model_name = 'ProGAN_Cyber'
    elif args == 'dogs':
        models_url = 'https://drive.google.com/uc?id=1-5VmBrlX8psMQhDyoMvTe_tLVUPSvAbe' # ProGAN_Dogs.zip
        imgs_url = 'https://drive.google.com/uc?id=1XMvtTtC6HLxjb3PGIeqQwWvQUHDcLqta' # dog_imgs.zip
        img_name = 'dog_imgs'
        model_name = 'ProGAN_Dogs'
    elif args == 'faces':
        models_url = 'https://drive.google.com/uc?id=1-7c9ro6Adfuy1T9gLpfmx1JRM2ARrEdA' # ProGAN_Faces.zip
        imgs_url = 'https://drive.google.com/uc?id=1-1bngZsB4n8_eXi94dSA2IiUNgpCijF_' # face_imgs.zip
        img_name = 'face_imgs'
        model_name = 'ProGAN_Faces'
    elif args == 'potato':
        models_url = 'https://drive.google.com/uc?id=1-BOOZAnZ6ecQPkHdndojjNglAqZtPZBW' # ProGAN_Potato.zip
        imgs_url = 'https://drive.google.com/uc?id=1A423Vi62SWb3FHtwieXDlmtIEcQHb2ub' # imgs.zip
        img_name = 'imgs'
        model_name = 'ProGAN_Potato'

    model_file = "models.zip"
    imgs_file = "imgs.zip"

    # download models if models folder does not exist
    if not os.path.exists(config.MODEL_PATH):
        gdown.download(models_url, model_file, quiet=False)

        with zipfile.ZipFile(model_file, 'r') as zip_ref:
            zip_ref.extractall()
        os.rename(model_name,"models")
        os.remove(model_file)

    # download logs if models folder does not exist
    if not os.path.exists(config.DATASET):
        gdown.download(imgs_url, imgs_file, quiet=False)

        with zipfile.ZipFile(imgs_file, 'r') as zip_ref:
            zip_ref.extractall()
        os.rename(img_name, "imgs")
        os.remove(imgs_file)

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

def apply_transform(args):
    '''yields a single real image batch to results folder'''
    if not os.path.exists(config.RESULTS): # check if results folder exists
        os.makedirs(config.RESULTS) # create results folder if does not exist

    loader, _ = get_loader(128)
    loop = tqdm(loader, leave=True)

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.DEVICE)
        for i in range(real.shape[0]):
            torchvision.utils.save_image(real[i, :, :, :]*0.5+0.5, Path(f"{config.RESULTS}/img_{batch_idx}_{i}.png"))
        if batch_idx == args:
            break

# CLI Driver ##
@click.command()
@click.argument('option', required=False)
@click.argument('args', required=False, nargs=-1)
def cli(args, option):
    commands = ['init', 'sample', 'download', 'removedups','transform']

    if not option:
        print(f'no option provided')
    elif option not in commands:
        print(f'invalid option: {option}')
    elif option == 'init':
        print(f'option: {option}')
        init()
    elif option == 'sample':
        if not args:
            args = (10,4)
        else:
            args = tuple(map(int, args))
        print(f'sample: {args}')
        generate_samples(args)
    elif option == 'download':
        if not args:
            args = ('cyber',)
        elif isinstance(args, str):
            args = args.lower().strip()
        else:
            print('please provide a valid option [cars, cyber, dogs, faces, potatoes]')
        print(f'option: {option}')
        download_models(args[0])
    elif option == 'removedups':
        print(f'option: {option}')
        remove_dups()
    elif option == 'transform':
        if not args:
            args = float('inf')
        else:
            args = tuple(map(int, args))[0]
        print(f'option: {option}')
        apply_transform(args)

if __name__ == "__main__":
    cli()