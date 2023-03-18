import os
import torch
import config
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from model import Model
from utils import model_test
from utils import make_directory
from utils import load_checkpoint
from utils import save_checkpoint
from utils import get_current_time
from utils import get_last_checkpoint
from torch.utils.data import DataLoader
from dataset import CrimeanPlantsDataset
from torch.utils.tensorboard import SummaryWriter


def train(model, opt, data_loader, num_epochs, current_epoch=0, writer=None, criterion=None):

    if writer is None:
        writer = SummaryWriter(f"./tb/train/{get_current_time()}")

    if criterion is None:
        # Finding the class weights:
        dataset_len = len(data_loader.dataset)
        class_value_counts = data_loader.dataset.data_csv["label"].value_counts(sort=False)
        class_weights = torch.Tensor([dataset_len / (x * config.OUT_FEATURES) for x in class_value_counts])
        class_weights = class_weights.to(config.DEVICE)

        # Define the Loss function and pass the class weights to it:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Finding the best (current) network accuracy:
    best_accuracy = model_test(model)

    # Learning cycle:
    for epoch in range(current_epoch + 1, num_epochs + 1):

        model.train()  # Switching the model to training mode

        epoch_loss = 0.0
        for idx, (images, labels) in enumerate(tqdm(data_loader)):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # Getting model predictions for the current batch:
            predictions = model(images).to(config.DEVICE)

            # Calculating the loss:
            loss = criterion(predictions, labels)
            epoch_loss += loss

            # Updating the model weights:
            opt.zero_grad()
            loss.backward()
            opt.step()

        # After each epoch we test the model:
        model.eval()  # Switching the model to evaluation mode
        current_accuracy = model_test(model)  # Testing

        # Updating tensorboard:
        writer.add_scalar("Accuracy", current_accuracy, global_step=epoch)  # Current accuracy
        writer.add_scalar("Loss", epoch_loss, global_step=epoch)  # Total loss for the current epoch

        # Updating the best accuracy:
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy

            # We save the checkpoint with the best accuracy, if necessary:
            if config.SAVE_BEST_MODEL:
                print("\033[32m=> Saving a checkpoint\033[0m")

                # Creating a directory to save
                dir_name = get_current_time()
                dir_path = os.path.join(config.CHECKPOINT_DIR, dir_name)
                make_directory(dir_path)

                # Save
                model_path = os.path.join(dir_path, config.CHECKPOINT_NAME)
                save_checkpoint(model, opt, model_path, epoch)

    writer.close()

    return best_accuracy


def main():
    # Initializing the model:
    model = Model(model_name=config.MODEL).to(config.DEVICE)

    # Parameters that we will update during training:
    params_to_update = [param for param in model.parameters() if param.requires_grad]

    # Initializing the optimizer:
    opt = optim.Adam(params=params_to_update, lr=config.LEARNING_RATE)

    # Loading the dataset:
    dataset = CrimeanPlantsDataset(
        root_dir=config.DATASET_DIR,
        csv_file=os.path.join(config.DATASET_DIR, "train_labels.csv"),
        transform=config.train_transforms,
        oversampling=True
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    current_epoch = 0

    # Loading the latest checkpoint:
    if config.LOAD_MODEL:
        print("\033[32m=> Loading the latest checkpoint\033[0m")

        checkpoint_path = get_last_checkpoint()
        model, opt, current_epoch = load_checkpoint(model, opt, checkpoint_path)

    num_epochs = config.NUM_EPOCHS

    # Training:
    train(model, opt, data_loader, num_epochs, current_epoch=current_epoch)


if __name__ == "__main__":
    main()
