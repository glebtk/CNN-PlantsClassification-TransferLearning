import os
import random
import shutil
import requests

from tqdm import tqdm
from dataset_preparation.parsing.yandex_images_parser import Parser


def find_images(plant: str, number: int, delay: float = 6.0, **kwargs) -> list:
    """Выполняет поиск картинок по запросу, возвращает список ссылок"""
    parser = Parser()
    delay = randomize_delay(delay)

    return parser.query_search(query=plant, limit=number, delay=delay, **kwargs)


def find_similar_images(url: str, number: int, delay: float = 6.0, **kwargs) -> list:
    """Выполняет поиск похожих изображений, возвращает список ссылок"""
    parser = Parser()
    delay = randomize_delay(delay)

    return parser.image_search(url=url, limit=number, delay=delay, **kwargs)


def make_directory(dir_path: str):
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)


def remove_duplicates(urls: list) -> list:
    """Удаляет дублирующиеся URL-ссылки из списка, возвращает список уникальных ссылок"""

    unique_urls = []
    for url in urls:
        if url not in unique_urls:
            unique_urls.append(url)

    return unique_urls


def save_images(urls: list, dir_path: str, prefix: str = "", number_images: bool = False):
    broken_url_counter = 0
    for i, url in enumerate(tqdm(urls)):
        image_name = prefix + str(url.split('/')[-1])

        if number_images:
            image_name = f"{i}_" + image_name

        path = os.path.join(dir_path, image_name)

        try:
            r = requests.get(url=url, allow_redirects=True, timeout=3.0)
            open(path, 'wb').write(r.content)
        except Exception:
            broken_url_counter += 1

    print(f"Сохранено изображений: {len(urls) - broken_url_counter}.\tНеудачно: {broken_url_counter}.\n")


def randomize_delay(delay: float) -> float:
    """Добавляет рандома в задержку. Возвращает полученное число измененное случайным образом на 15%."""
    return delay * random.uniform(0.85, 1.15)
