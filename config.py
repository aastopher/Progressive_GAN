import cv2
import torch
from math import log2
import os
from pathlib import Path

CWD = os.getcwd() # current working directory

START_TRAIN_AT_IMG_SIZE = 4
DATASET = Path(f"{CWD}/imgs")
RESULTS = Path(f"{CWD}/results")
CURRENT_IMG_SIZE = Path(f"{CWD}/models/img_size.dat")
CHECKPOINT_GEN = Path(f"{CWD}/models/generator.pth")
CHECKPOINT_CRITIC = Path(f"{CWD}/models/critic.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = True
LEARNING_RATE = 1e-3
BATCH_SIZES = [32, 32, 32, 16, 16, 16, 16, 8, 4]
CHANNELS_IMG = 3
Z_DIM = 256  # should be 512 in original paper
IN_CHANNELS = 256  # should be 512 in original paper
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4