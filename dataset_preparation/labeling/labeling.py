import os
import random
import shutil
import pandas as pd

from labeling_utils import make_directory


def labeling(dataset_dir: str, remove_source: bool = False):
    # Переименовываем директорию датасета:
    source_dir = "data/dataset_source"
    os.rename(dataset_dir, source_dir)

    # Создаём новые директории для сохранения результата:
    make_directory(dataset_dir)

    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    make_directory(train_dir)
    make_directory(test_dir)

    train_labels = pd.DataFrame(columns=["path", "label"])
    test_labels = pd.DataFrame(columns=["path", "label"])

    dirs = os.listdir(source_dir)

    # Проходимся циклом по всем директориям с изображениями:
    for label, dir_ in enumerate(dirs):
        current_dir_path = os.path.join(source_dir, dir_)

        if not os.path.isdir(current_dir_path):
            continue

        print(f"\033[32m\033[1mОбработка:\033[0m {label+1}. {dir_}")

        # Создаём в train и test соответствующие директории:
        make_directory(os.path.join(train_dir, dir_))
        make_directory(os.path.join(test_dir, dir_))

        label_name = dir_.lower().replace(" ", "_").replace("-", "_")

        image_names = os.listdir(current_dir_path)  # Имена изображений в текущей директории
        random.shuffle(image_names)  # Перемешиваем имена изображений

        train_len = int(len(image_names) * 0.85)  # Вычисляем длину обучающей выборки. Она будет составлять 85%

        # Разделяем на test и train:
        train_image_names = image_names[:train_len]
        test_image_names = image_names[train_len:]

        # Проходимся циклом по обучающей выборке:
        for number, name in enumerate(train_image_names):
            new_name = label_name + "_train_" + str(number + 1) + ".jpg"  # Новое имя изображения

            old_path = os.path.join(current_dir_path, name)  # Текущий путь к изображению
            new_path = os.path.join(*[train_dir, dir_, new_name])  # Новый путь изображения

            # Добавляем запись в csv-фaйл:
            record = pd.DataFrame({"path": [f"train/{dir_}/{new_name}"], "label": [label]})
            train_labels = pd.concat([train_labels, record])

            # Перемещаем и переименовываем изображение:
            shutil.move(old_path, new_path)

        # Проходимся циклом по тестовой выборке:
        for number, name in enumerate(test_image_names):
            new_name = label_name + "_test_" + str(number + 1) + ".jpg"  # Новое имя изображения

            old_path = os.path.join(current_dir_path, name)  # Текущий путь к изображению
            new_path = os.path.join(*[test_dir, dir_, new_name])  # Новый путь изображения

            # Добавляем запись в csv-фaйл:
            record = pd.DataFrame({"path": [f"test/{dir_}/{new_name}"], "label": [label]})
            test_labels = pd.concat([test_labels, record])

            # Перемещаем и переименовываем изображение:
            shutil.move(old_path, new_path)

    # Если требуется, удаляем изначальную директорию:
    if remove_source:
        shutil.rmtree(source_dir)

    # Сохраняем csv-файлы:
    train_labels.to_csv("data/dataset/train_labels.csv", index=False)
    test_labels.to_csv("data/dataset/test_labels.csv", index=False)

    print("\033[32m\033[1m\n\t...Готово!\033[0m")


if __name__ == "__main__":
    labeling("data/dataset/", remove_source=False)
