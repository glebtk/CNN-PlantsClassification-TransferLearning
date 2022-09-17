import torch
import numpy as np
import torchvision.transforms as transforms

# Предустановки
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 64
IN_CHANNELS = 3
OUT_CHANNELS = 50
NUM_WORKERS = 2

# Обучение
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 3e-4

LOAD_MODEL = False
SAVE_MODEL = False
USE_TENSORBOARD = True

# Датасет
DATASET_DIR = "./data/dataset"
DATASET_MEAN = np.array([0.4056, 0.4354, 0.2909])
DATASET_STD = np.array([0.2074, 0.1983, 0.2063])

# Другое
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_NAME = "plants_model.pth.tar"


train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=1.85, p=0.15),
        transforms.RandomRotation(degrees=(-45, 45)),
        transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.25, 1.0)),
        transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
     ]
)

test_transforms = transforms.Compose(
    [
        transforms.CenterCrop(size=256),
        transforms.Resize(size=IMAGE_SIZE),
        transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
     ]
)
