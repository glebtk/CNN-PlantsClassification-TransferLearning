import os
import zipfile
import urllib.request


# Предустановки
LOAD_DATASET = True         # Загрузить датасет для обучения
LOAD_CHECKPOINT = True      # Загрузить чекпоинт с предобученными весами


def download_and_unzip(url, path, name):
    full_path = os.path.join(path, name)

    urllib.request.urlretrieve(url, full_path)

    dataset_zip = zipfile.ZipFile(full_path, 'r')
    dataset_zip.extractall(path)


def download_files():
    if LOAD_DATASET:
        url = "https://gitlab.com/glebtutik/crimean_plants_classification_files/-/raw/main/data/dataset.zip"
        path = "data"
        file_name = "dataset.zip"

        download_and_unzip(url, path, file_name)
        os.remove(os.path.join(path, file_name))

        print("=> Датасет загружен!")

    if LOAD_CHECKPOINT:
        url = "https://gitlab.com/glebtutik/crimean_plants_classification_files/-/raw/main/checkpoints/plants_model_checkpoint.zip"
        path = "checkpoints"
        file_name = "plants_model_checkpoint.zip"

        download_and_unzip(url, path, file_name)
        os.remove(os.path.join(path, file_name))

        print("=> Чекпоинт загружен!")


if __name__ == "__main__":
    download_files()
