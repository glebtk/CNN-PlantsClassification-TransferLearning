import os
import config
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Model
from utils import model_test
from utils import make_directory
from utils import load_checkpoint
from utils import save_checkpoint
from utils import get_current_time
from utils import get_last_checkpoint
from dataset import CrimeanPlantsDataset


def train():
    # Инициализируем модель:
    model = Model(in_channels=config.IN_CHANNELS, out_channels=config.OUT_CHANNELS).to(config.DEVICE)

    # Инициализируем оптимизатор:
    opt = optim.Adam(
        params=list(model.parameters()),
        lr=config.LEARNING_RATE
    )

    # Загружаем датасет:
    dataset = CrimeanPlantsDataset(
        root_dir=config.DATASET_DIR,
        csv_file=os.path.join(config.DATASET_DIR, "train_labels.csv"),
        transform=config.train_transforms
    )

    data_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    # Загружаем последний чекпоинт модели:
    if config.LOAD_MODEL:
        print("\033[32m=> Загрузка последнего чекпоинта\033[0m")
        checkpoint_path = get_last_checkpoint()
        model, opt, current_epoch = load_checkpoint(model, opt, checkpoint_path)
    else:
        current_epoch = 0

    criterion = nn.CrossEntropyLoss()  # может быть надо добавить параметр weight (for unbalanced classes)

    writer = SummaryWriter(f"/tb/train/{get_current_time()}")

    # ----- Цикл обучения ----- #
    for epoch in range(current_epoch, config.NUM_EPOCHS):

        total_loss = 0.0
        for idx, (images, labels) in enumerate(tqdm(data_loader)):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # Получаем предсказания модели для текущего батча:
            predictions = model(images)

            # Вычисляем loss:
            loss = criterion(predictions, labels)
            total_loss += loss

            # Выполняем backpropagation:
            opt.zero_grad()
            loss.backward()
            opt.step()

            # if idx % 1 == 0:
            #     print(f"\n{idx}.goal:\t\t{torch.argmax(labels[0])}")
            #     print(f"{idx}.prediction:\t\t{torch.argmax(predictions[0])}")
            #     print(f"{idx}.{torch.softmax(predictions[0], dim=0)}")

        # После каждой эпохи тестируем модель:
        accuracy = model_test(model)

        # Обновляем tensorboard или выводим в консоль данные о процессе обучения:
        if config.USE_TENSORBOARD:
            writer.add_scalar("Accuracy", accuracy, global_step=epoch)
            writer.add_scalar("Loss", total_loss, global_step=epoch)
        else:
            print(f"Accuracy: {accuracy}\tLoss: {total_loss}\n")

        # Сохраняем модель:
        if config.SAVE_MODEL:
            print("\033[32m=> Сохранение чекпоинта\033[0m")

            # Создаем директорию для сохранения
            dir_path = os.path.join(config.CHECKPOINT_DIR, get_current_time())
            make_directory(dir_path)

            # Сохраняем
            model_path = os.path.join(dir_path, config.CHECKPOINT_NAME)
            save_checkpoint(model, opt, model_path, epoch)


if __name__ == "__main__":
    train()
