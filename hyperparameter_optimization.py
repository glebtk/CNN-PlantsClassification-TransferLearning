import os
import config
import optuna
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Model
from utils import model_test
from utils import make_directory
from utils import save_checkpoint
from dataset import CrimeanPlantsDataset


def tryn(model_name, learning_rate, batch_size, num_epochs=None, writer=None):

    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS

    if writer is None:
        writer = SummaryWriter(f"./tb/optim/mn={model_name}_lr={round(learning_rate, 5)}_bs={batch_size}")

    model = Model(model_name=model_name).to(config.DEVICE)

    # Инициализируем оптимизатор:
    opt = optim.Adam(
        params=list(model.parameters()),
        lr=learning_rate
    )

    # Загружаем датасет:
    dataset = CrimeanPlantsDataset(
        root_dir=config.DATASET_DIR,
        csv_file=os.path.join(config.DATASET_DIR, "train_labels.csv"),
        transform=config.train_transforms
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    for epoch in range(num_epochs):

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
        writer.add_scalar("Accuracy", current_accuracy, global_step=epoch)
        writer.add_scalar("Loss", epoch_loss, global_step=epoch)

        # Сохраняем чекпоинт, если необходимо:
        if config.SAVE_BEST_MODEL and current_accuracy > best_accuracy:
            # Обновляем точность:
            best_accuracy = current_accuracy

            # Создаем директорию для сохранения чекпоинта:
            dir_name = f"mn_{model_name}_lr_{round(learning_rate, 5)}_bs_{batch_size}_e_{epoch}"
            dir_path = os.path.join(config.CHECKPOINT_DIR, dir_name)
            make_directory(dir_path)

            # Сохраняем:
            model_path = os.path.join(dir_path, config.CHECKPOINT_NAME)
            save_checkpoint(model, opt, model_path, epoch)

    writer.close()

    return best_accuracy


def objective(trial):
    model_name = trial.suggest_categorical("model_name", ["resnet18", "alexnet", "mobilenet_v3_small"])
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2)
    batch_size = trial.suggest_int("batch_size", 1, 128)

    accuracy = tryn(model_name=model_name, learning_rate=lr, batch_size=batch_size, num_epochs=1)

    return accuracy


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    trial = study.best_trial
    print("Best Score: ", trial.value)
    print("Best Params: ")

    for key, value in trial.params.items():
        print("  {}: {}".format(key, value))


if __name__ == "__main__":
    main()
