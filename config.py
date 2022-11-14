import cv2
import torch
from math import log2
import os
from pathlib import Path
import numpy as np

CWD = os.getcwd() # current working directory

START_TRAIN_AT_IMG_SIZE = 4
DATASET = Path(f"{CWD}/imgs") # dataset directory
SAMPLE_SIZE = 10000 # dataloader sample size
RESULTS = Path(f"{CWD}/results") # results directory
MODEL_PATH = Path(f"{CWD}/models") # model directory
LOGS_PATH = Path(f"{CWD}/logs") # logs directory
CURRENT_IMG_SIZE = Path(f"{CWD}/models/img_size.dat") # image size store
CHECKPOINT_GEN = Path(f"{CWD}/models/generator.pth") # generator store
CHECKPOINT_CRITIC = Path(f"{CWD}/models/critic.pth") # critic store
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # init available device
SAVE_MODEL = True
LOAD_MODEL = True
LEARNING_RATE = 1e-3
# BATCH_SIZES = [32, 32, 32, 16, 16, 8, 8, 4] # tune for my GPU - 512
# BATCH_SIZES = [32, 32, 32, 16, 16, 8, 8] # tune for my GPU - 256
BATCH_SIZES = [32, 32, 32, 16, 16, 8] # tune for my GPU - 128
BATCH_REPEAT = 1 # number of times critic, critiques the batch
CHANNELS_IMG = 3
Z_DIM = 256  # 512 in original paper, 256 reduces vram usage
IN_CHANNELS = 256  # 512 in original paper, 256 reduces vram usage
LAMBDA_GP = 10
# PROGRESSIVE_EPOCHS = [195, 195, 195, 390, 390, 780] # 2046 * 3 
PROGRESSIVE_EPOCHS = [0, 0, 0, 0, 0, 334] # total steps should be 2145 but power went out so we need to run exactly 334 to hit 2145 from 1811; 1811 + 334 = 2145
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4 # how many sub-processes to use for data loading