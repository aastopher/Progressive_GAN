from cmath import inf
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils import (
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    get_loader)
from model import Discriminator, Generator
from math import log2
from tqdm import tqdm
import config
import time
import fid

torch.backends.cudnn.benchmarks = True

def train_fn(critic, gen, loader, dataset, step, alpha, opt_critic, opt_gen, tensorboard_step, writer, scaler_gen, scaler_critic, fid, img_size):
    """ Training of ProGAN using WGAN-GP loss"""
    loop = tqdm(loader, leave=True)
    for batch_idx, (real, _) in enumerate(loop): # loop through dataset
        for _ in range(config.BATCH_REPEAT): # number of times batch is seen by the critic network
            real = real.to(config.DEVICE)
            cur_batch_size = real.shape[0]

            # Train Critic: max E[critic(real)] - E[critic(fake)] <-> min -E[critic(real)] + E[critic(fake)]
            # which is equivalent to minimizing the negative of the expression
            noise = torch.randn(cur_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)

            with torch.cuda.amp.autocast():
                fake = gen(noise, alpha, step) # generate fakes
                critic_real = critic(real, alpha, step) # critique the reals
                critic_fake = critic(fake.detach(), alpha, step) # critique the fakes
                gp = gradient_penalty(critic, real, fake, alpha, step, device=config.DEVICE) # compute gradient penalty

                # compute wasserstein-gp loss
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + config.LAMBDA_GP * gp
                    + (0.001 * torch.mean(critic_real ** 2)))

            # update the critic network
            opt_critic.zero_grad()
            scaler_critic.scale(loss_critic).backward()
            scaler_critic.step(opt_critic)
            scaler_critic.update()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            with torch.cuda.amp.autocast():
                gen_fake = critic(fake, alpha, step) # critique the fakes
                loss_gen = -torch.mean(gen_fake) # compute the generator loss

            # update the generator network
            opt_gen.zero_grad()
            scaler_gen.scale(loss_gen).backward()
            scaler_gen.step(opt_gen)
            scaler_gen.update()

            # update alpha and ensure less than 1
            alpha += cur_batch_size / (
                (config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset)
            )
            alpha = min(alpha, 1)

            # occasionally plot to tensorboard
            if batch_idx % 500 == 0:

                with torch.no_grad():
                    fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
                plot_to_tensorboard(
                    writer,
                    loss_critic.item(),
                    loss_gen.item(),
                    fid,
                    real.detach(),
                    fixed_fakes.detach(),
                    img_size,
                    tensorboard_step,
                )
                tensorboard_step += 1

            loop.set_postfix(
                gp=gp.item(),
                loss_critic=loss_critic.item(),
                fid=fid,
            )

    return tensorboard_step, alpha, real, fake


def main():
    # initialize gen and disc, note: discriminator should be called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])
    # but really who cares..
    gen = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
    critic = Discriminator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)

    # initialize optimizers and scalers for FP16 (floating point 16 or half precision) training
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    # init the FID
    block_idx = fid.InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    fid_model = fid.InceptionV3([block_idx])
    fid_model = fid_model.cuda()

    # for tensorboard plotting
    writer = SummaryWriter(f"logs/{time.strftime('%Y%m%d-%H%M%S')}")

    # load model according to config.py
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,) # load generator
        load_checkpoint(config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE,) # load critic
        current_image_size = int(config.CURRENT_IMG_SIZE.read_text()) # collect current image size
    else:
        current_image_size = config.START_TRAIN_AT_IMG_SIZE # set to starting size according to config.py

    # set networks to train mode explicitly
    gen.train()
    critic.train()

    tensorboard_step = 0 # init tensorboard steps

    # init step at img size according to config.py
    step = int(log2(current_image_size / 4))
    fretchet_dist = inf # init fid
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]: # number of epochs (progressive) loop as defined by config.py
        alpha = 1e-5  # start with very low alpha
        img_size = 4 * 2 ** step # compute image size
        loader, dataset = get_loader(img_size)  # 4->0, 8->1, 16->2, 32->3, 64->4, 128->5, 256->6, 512->7, 1024->8
        print(f"Current image size: {img_size}")
        print(f"{img_size}", file=open(config.CURRENT_IMG_SIZE, 'w'))

        # define epoch steps
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            tensorboard_step, alpha, real, fake = train_fn(critic, gen, loader, dataset, step, alpha, opt_critic, opt_gen, tensorboard_step, writer, scaler_gen, scaler_critic, fretchet_dist, img_size)

            # save model as defined by config.py
            if config.SAVE_MODEL:
                save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(critic, opt_critic, filename=config.CHECKPOINT_CRITIC)

        fretchet_dist = fid.calculate_fretchet(real, fake, fid_model) # calculate the FID at end of step
        step += 1  # progress to the next img size 
    writer.add_scalar("FID", fretchet_dist, global_step=tensorboard_step) # add FID for end of training

if __name__ == "__main__":
    main()
