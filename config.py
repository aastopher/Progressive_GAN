import cv2
import torch
from math import log2
import os
from pathlib import Path
import numpy as np

CWD = os.getcwd() # current working directory

START_TRAIN_AT_IMG_SIZE = 4
DATASET = Path(f"{CWD}/imgs")
SAMPLE_SIZE = 10000 # dataloader sample size
RESULTS = Path(f"{CWD}/results")
MODEL_PATH = Path(f"{CWD}/models")
LOGS_PATH = Path(f"{CWD}/logs")
CURRENT_IMG_SIZE = Path(f"{CWD}/models/img_size.dat")
CHECKPOINT_GEN = Path(f"{CWD}/models/generator.pth")
CHECKPOINT_CRITIC = Path(f"{CWD}/models/critic.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = True
LEARNING_RATE = 1e-3
# BATCH_SIZES = [32, 32, 32, 16, 16, 8, 8, 4] # tune for my GPU - 512
# BATCH_SIZES = [32, 32, 32, 16, 16, 8, 8] # tune for my GPU - 256
BATCH_SIZES = [32, 32, 32, 16, 16, 8] # tune for my GPU - 128
BATCH_REPEAT = 1
CHANNELS_IMG = 3
Z_DIM = 256  # 512 in original paper
IN_CHANNELS = 256  # 512 in original paper
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
# PROGRESSIVE_EPOCHS = [150] * len(BATCH_SIZES)
# PROGRESSIVE_EPOCHS = list(np.linspace(65, 256, len(BATCH_SIZES), dtype=int))
# PROGRESSIVE_EPOCHS = [325, 325, 325, 650, 650, 1300] # 2046 * 5
# PROGRESSIVE_EPOCHS = [260, 260, 260, 520, 520, 1040] # 2046 * 4
# PROGRESSIVE_EPOCHS = [195, 195, 195, 390, 390, 780] # 2046 * 3 
PROGRESSIVE_EPOCHS = [0, 0, 0, 0, 0, 334] # total steps should be 2145 but power went out so we need to run exactly 334 to hit 2145 from 1811; 1811 + 334 = 2145
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4