import torch
import torchvision.transforms as transforms

# Предустановки
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "mobilenet_v3_small"
IMAGE_SIZE = 224
IN_CHANNELS = 3
OUT_FEATURES = 50
NUM_WORKERS = 2

# Обучение
NUM_EPOCHS = 100
BATCH_SIZE = 127
LEARNING_RATE = 0.00171

LOAD_MODEL = False
SAVE_BEST_MODEL = True

# Датасет
DATASET_DIR = "./data/dataset"

# Mean и std imagenet:
MEAN = torch.Tensor([0.485, 0.456, 0.406])
STD = torch.Tensor([0.229, 0.224, 0.225])

# Mean и std датасета растений:
# MEAN = torch.Tensor([0.4056, 0.4354, 0.2909])
# STD = torch.Tensor([0.2074, 0.1983, 0.2063])

# Другое
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_NAME = "plants_model.pth.tar"

train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9),
        transforms.RandomAdjustSharpness(sharpness_factor=1.85, p=0.15),
        transforms.RandomRotation(degrees=(-45, 45)),
        transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.15, 2.0)),
        transforms.Normalize(mean=MEAN, std=STD),
     ]
)

test_transforms = transforms.Compose(
    [
        transforms.CenterCrop(size=256),
        transforms.Resize(size=IMAGE_SIZE),
        transforms.Normalize(mean=MEAN, std=STD),
     ]
)
