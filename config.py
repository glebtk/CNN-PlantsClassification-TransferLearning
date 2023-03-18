import torch
import torchvision.transforms as transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "mobilenet_v3_large"
IMAGE_SIZE = 224
IN_CHANNELS = 3
OUT_FEATURES = 50
NUM_WORKERS = 2

# Training
NUM_EPOCHS = 100
BATCH_SIZE = 90
LEARNING_RATE = 9e-05

LOAD_MODEL = False
SAVE_BEST_MODEL = True

# Dataset
DATASET_DIR = "./data/dataset"

# Mean and std (imagenet):
MEAN = torch.Tensor([0.485, 0.456, 0.406])
STD = torch.Tensor([0.229, 0.224, 0.225])

# Other
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_NAME = "plants_model.pth.tar"

train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.9, contrast=0.7, saturation=0.9),
        transforms.RandomAdjustSharpness(sharpness_factor=1.85, p=0.15),
        transforms.RandomRotation(degrees=(-45, 45)),
        transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
        transforms.RandomResizedCrop(size=IMAGE_SIZE,
                                     scale=(0.2, 1.0),
                                     interpolation=transforms.InterpolationMode.NEAREST),
        transforms.RandomErasing(p=0.3, scale=(0.01, 0.2)),
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
