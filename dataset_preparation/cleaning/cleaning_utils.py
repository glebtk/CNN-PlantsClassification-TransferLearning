import os
import shutil
import imagehash

from PIL import Image
from PIL import ImageFile


def make_directory(dir_path: str):
    """
    Описание
    ---------
    Создаёт новую директорию. Если директория уже существует, перезаписывает.

    Параметры
    ---------
    **dir_path:** str
        Путь, по которому будет создана директория
    """

    try:
        os.makedirs(dir_path)
    except FileExistsError:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)


def verify_image(path: str) -> bool:
    """
    Описание
    ---------
    Проверяет, является ли файл действительным изображением.

    Параметры
    ---------
    **path:** str
        Путь к изображению

    Возвращаемое значение
    ---------
    bool: Значение, бинарным способом описывающее, является ли файл изображением
    """

    try:
        Image.open(path)
        return True
    except Exception:
        return False


def check_filters(path: str, min_size: int = 1, mode: str = "RGB") -> bool:
    """
    Описание
    ---------
    Проверяет изображение на соответствие фильтрам.

    Параметры
    ---------
    **path:** str
        Путь к изображению
    **min_size:** int
        Минимальный размер изображения по меньшей стороне
    **mode:** str
        Цветовой режим

    Возвращаемое значение
    ---------
    bool: Значение соответствия (или несоответствия) изображения фильтрам
    """

    image = Image.open(path)
    if min(image.size) < min_size or image.mode != mode:
        return False
    else:
        return True


def remove_duplicates(paths: list, epsilon: int = 1) -> list:
    """
    Описание
    ---------
    Удаляет из списка пути к одинаковым или похожим изображениям.

    Параметры
    ---------
    **paths:** list
        Список путей к изображениям
    **epsilon:** int
        Степень сходства между изображениями

    Возвращаемое значение
    ---------
    list: Список путей к уникальным изображениям
    """

    ImageFile.LOAD_TRUNCATED_IMAGES = True  # Позволяет обрабатывать поврежденные изображения

    hashes = [imagehash.average_hash(Image.open(path)) for path in paths]

    path_hash = [list(tup) for tup in zip(paths, hashes)]
    path_hash = sorted(path_hash, key=lambda item: str(item[1]))

    unique_images = []
    for ph in path_hash:
        if len(unique_images) == 0 or ph[1] - unique_images[-1][1] > epsilon:
            unique_images.append(ph)

    unique_images = [item[0] for item in unique_images]

    return unique_images


def resize_image(image: Image, size: int) -> Image:
    """
    Описание
    ---------
    Изменяет размер изображения.

    Параметры
    ---------
    **image:** Image
        PIL-изображениe.
    **size:** int
        Целевой размер изображения по меньшей стороне (px)

    Возвращаемое значение
    ---------
    Image: Отмасштабированное изображение.
    """

    factor = size / min(image.size)
    new_size = (int(image.width * factor), int(image.height * factor))
    return image.resize(new_size)


def resize_images(images: list, size: int) -> list:
    """
    Описание
    ---------
    Изменяет размер изображений в списке.

    Параметры
    ---------
    **images:** list
        Список PIL-изображений.
    **size:** int
        Целевой размер изображений по меньшей стороне (px)

    Возвращаемое значение
    ---------
    list: Список отмасштабированных изображений
    """
    resized_images = []
    for image in images:
        factor = size / min(image.size)
        new_size = (int(image.width * factor), int(image.height * factor))
        resized_images.append(image.resize(new_size))

    return resized_images


def save_images(images: list, dir_path: str):
    """
    Описание
    ---------
    Создаёт новую директорию по заданному пути, и сохраняет в нее изображения.

    Параметры
    ---------
    **images:** list
        Список PIL-изображений.
    **dir_path:** str
        Путь к директории
    """
    make_directory(dir_path)
    for i, image in enumerate(images):
        img_name = f"{i}.jpg"
        img_path = os.path.join(dir_path, img_name)
        image.save(img_path)
