import os
import config
import optuna
import torch.optim as optim

from train import train
from model import Model
from torch.utils.data import DataLoader
from dataset import CrimeanPlantsDataset
from torch.utils.tensorboard import SummaryWriter


def objective(trial):
    models = ["alexnet", "convnext_tiny", "densenet121", "densenet201", "resnet18", "mobilenet_v3_small", "mobilenet_v3_large"]

    model_name = trial.suggest_categorical("model_name", models)
    learning_rate = trial.suggest_float("learning_rate", 3e-5, 1e-2),
    batch_size = trial.suggest_int("batch_size", 16, 128)

    writer = SummaryWriter(f"./tb/optim/mn={model_name}_lr={round(learning_rate[0], 5)}_bs={batch_size}")

    model = Model(model_name=model_name).to(config.DEVICE)

    # Определяем параметры, которые будем обновлять при обучении:
    params_to_update = [param for param in model.parameters() if param.requires_grad]

    # Инициализируем оптимизатор:
    opt = optim.Adam(params=params_to_update, lr=learning_rate[0])

    # Загружаем датасет:
    dataset = CrimeanPlantsDataset(
        root_dir=config.DATASET_DIR,
        csv_file=os.path.join(config.DATASET_DIR, "train_labels.csv"),
        transform=config.train_transforms,
        oversampling=True
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    accuracy = train(model, opt, data_loader, num_epochs=5, writer=writer)

    return accuracy


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    trial = study.best_trial
    print("Лучшая точность: ", trial.value)
    print("Лучшие гиперпараметры: ")

    for key, value in trial.params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
