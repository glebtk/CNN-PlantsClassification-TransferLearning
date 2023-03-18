import os
import sys
import torch
import random
import shutil
import config
import numpy as np

from datetime import datetime
from torch.utils.data import DataLoader
from dataset import CrimeanPlantsDataset


def make_directory(dir_path: str) -> None:
    """Creates a directory. If the directory exists, it overwrites it."""

    try:
        os.makedirs(dir_path)
    except FileExistsError:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)


def save_checkpoint(model, optimizer, model_path, epoch=0) -> None:
    """Saves the checkpoint in the learning process (model, optimizer, epoch number)."""

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, model_path)


def load_checkpoint(model, optimizer, checkpoint_file):
    """Loads the checkpoint of the model. Returns model, optimizer, epoch number"""

    try:
        checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]

        return model, optimizer, epoch

    except FileNotFoundError:
        print(f"Error: couldn't find {checkpoint_file}")
        sys.exit(1)


def get_last_checkpoint() -> str:
    """Returns the path to the last saved checkpoint."""
    try:
        checkpoints = os.listdir(config.CHECKPOINT_DIR)
        checkpoints = [os.path.join(config.CHECKPOINT_DIR, d) for d in checkpoints]
        checkpoints = [d for d in checkpoints if os.path.isdir(d)]
        checkpoints.sort(key=lambda x: os.path.getmtime(x))  # Сортировка по времени

        path_to_model = os.path.join(checkpoints[-1], config.CHECKPOINT_NAME)

        return path_to_model
    except IndexError:
        print(f"Error: there are no saved checkpoints in the {config.CHECKPOINT_DIR} directory")
        sys.exit(1)
    except FileNotFoundError:
        print(f'Error: failed to load {config.CHECKPOINT_NAME}')
        sys.exit(1)


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def model_test(model) -> float:
    """Performs model testing on a test data. Returns accuracy."""

    model = model.to(config.DEVICE)
    model.eval()

    # Load dataset
    dataset = CrimeanPlantsDataset(
        root_dir=config.DATASET_DIR,
        csv_file=os.path.join(config.DATASET_DIR, "test_labels.csv"),
        transform=config.test_transforms
    )

    data_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    with torch.no_grad():
        correct = 0  # Counter of correct answers

        # Cycle through all the batches in the test sample:
        for images, labels in data_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # Getting predictions on current images:
            predictions = model(images)


            for prediction, label in zip(predictions, labels):
                # If the prediction is correct,
                if torch.argmax(torch.softmax(prediction, dim=0)) == label:
                    correct += 1  # we count the correct answer

    # Accuracy = (number of correct answers) / (total number of images):
    accuracy = correct / len(dataset)

    return accuracy


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    print(f"Random seed set as {seed}")
