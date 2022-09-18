import os
import sys
import torch
import shutil
import config

from datetime import datetime
from torch.utils.data import DataLoader
from dataset import CrimeanPlantsDataset


def make_directory(dir_path: str):
    """Создаёт директорию. Если директория существует - перезаписывает."""

    try:
        os.makedirs(dir_path)
    except FileExistsError:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)


def save_checkpoint(model, optimizer, model_path, epoch=0):
    """Сохраняет чекпоинт модели в процессе обучения (модель, оптимизатор, номер эпохи)."""

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, model_path)


def load_checkpoint(model, optimizer, checkpoint_file):
    """Загружает чекпоинт модели. Возвращает модель, оптимизатор, номер эпохи"""

    try:
        checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]

        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = lr

        return model, optimizer, epoch

    except FileNotFoundError:
        print(f"Ошибка: не удалось найти {checkpoint_file}")
        sys.exit(1)


def get_last_checkpoint():
    """Возвращает путь к последнему по времени сохранённому чекпоинту."""
    try:
        checkpoints = os.listdir(config.CHECKPOINT_DIR)
        checkpoints = [os.path.join(config.CHECKPOINT_DIR, d) for d in checkpoints]
        checkpoints = [d for d in checkpoints if os.path.isdir(d)]
        checkpoints.sort(key=lambda x: os.path.getmtime(x))  # Сортировка по времени

        path_to_model = os.path.join(checkpoints[-1], config.CHECKPOINT_NAME)

        return path_to_model
    except IndexError:
        print(f"Ошибка: в директории {config.CHECKPOINT_DIR} нет сохраненных чекпоинтов")
        sys.exit(1)
    except FileNotFoundError:
        print(f'Ошибка: не удалось загрузить {config.CHECKPOINT_NAME}')
        sys.exit(1)


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def model_test(model) -> float:
    """Проводит тестирование модели на тестовой выборке. Возвращает точность (accuracy)."""

    model = model.to(config.DEVICE)

    # Загружаем датасет
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
        correct = 0  # Счётчик правильных ответов

        # Проходимся циклом по всем батчам в тестовой выборке:
        for images, labels in data_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            predictions = model(images)  # Получаем предсказания на текущих изображениях

            for prediction, label in zip(predictions, labels):
                # Если предсказание правильное,
                if torch.argmax(torch.softmax(prediction, dim=0)) == torch.argmax(label):
                    correct += 1  # засчитываем правильный ответ

    accuracy = correct / len(dataset)  # Точность = количество правильных ответов / общее количество изображений

    return accuracy
