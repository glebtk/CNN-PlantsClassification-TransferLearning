import os

import torch

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


def train(model, opt, data_loader, num_epochs, current_epoch=0, writer=None, criterion=None):

    if writer is None:
        writer = SummaryWriter(f"./tb/train/{get_current_time()}")

    if criterion is None:
        dataset_len = len(data_loader.dataset)  # Длина датасета
        class_value_counts = data_loader.dataset.data_csv["label"].value_counts(sort=False)  # Кол-во изображений каждого класса
        class_weights = torch.Tensor([1-x/dataset_len for x in class_value_counts])  # Веса классов

        criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_accuracy = 0.0
    for epoch in range(current_epoch, num_epochs):

        model.train()  # Переключение модели в режим обучения

        epoch_loss = 0.0
        for idx, (images, labels) in enumerate(tqdm(data_loader)):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # Получаем предсказания модели для текущего батча:
            predictions = model(images)

            # Вычисляем loss:
            loss = criterion(predictions, labels)
            epoch_loss += loss

            # Обновляем веса модели:
            opt.zero_grad()
            loss.backward()
            opt.step()

        # После каждой эпохи тестируем модель:
        model.eval()  # Переключение модели в режим тестирования
        current_accuracy = model_test(model)  # Тестирование

        # Обновляем tensorboard:
        writer.add_scalar("Accuracy", current_accuracy, global_step=epoch)  # Текущая точность модели
        writer.add_scalar("Loss", epoch_loss, global_step=epoch)  # Суммарный loss за текущую эпоху

        # Сохраняем чекпоинт, если необходимо:
        if config.SAVE_BEST_MODEL and current_accuracy > best_accuracy:
            print("\033[32m=> Сохранение чекпоинта\033[0m")

            # Обновляем лучшую точность:
            best_accuracy = current_accuracy

            # Создаем директорию для сохранения
            dir_name = get_current_time()
            dir_path = os.path.join(config.CHECKPOINT_DIR, dir_name)
            make_directory(dir_path)

            # Сохраняем
            model_path = os.path.join(dir_path, config.CHECKPOINT_NAME)
            save_checkpoint(model, opt, model_path, epoch)

    writer.close()


def main():
    # Инициализируем модель:
    model = Model(model_name=config.MODEL).to(config.DEVICE)

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
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    current_epoch = 0  # Текущая эпоха обучения

    # Загружаем последний чекпоинт модели:
    if config.LOAD_MODEL:
        print("\033[32m=> Загрузка последнего чекпоинта\033[0m")

        checkpoint_path = get_last_checkpoint()
        model, opt, current_epoch = load_checkpoint(model, opt, checkpoint_path)

    num_epochs = config.NUM_EPOCHS  # Количество эпох обучения

    # Обучаем модель:
    train(model, opt, data_loader, num_epochs, current_epoch=current_epoch)


if __name__ == "__main__":
    main()
