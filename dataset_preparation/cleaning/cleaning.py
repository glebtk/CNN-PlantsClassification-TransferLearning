import os

from PIL import Image

from cleaning_utils import verify_image
from cleaning_utils import check_filters
from cleaning_utils import remove_duplicates
from cleaning_utils import resize_image
from cleaning_utils import save_images


def main():
    images_dir = "./data/images"
    dataset_dir = "./data/dataset"

    source_dir_names = os.listdir(images_dir)

    for dir_name in source_dir_names:
        source_dir = os.path.join(images_dir, dir_name)

        if os.path.isdir(source_dir):
            print(f"\033[32m\033[1mОбработка:\033[0m {dir_name}.")

            images_names = os.listdir(source_dir)  # Список имён изображений в директории

            # Получаем пути к изображениям:
            paths_to_images = [os.path.join(source_dir, name) for name in images_names]

            # Проверяем, лежат ли по путям именно изображения:
            paths_to_images = [path for path in paths_to_images if verify_image(path)]

            # Проверяем, удовлетворяют ли эти изображения условиям (размер, цветовой режим):
            paths_to_images = [path for path in paths_to_images if check_filters(path, min_size=256, mode="RGB")]

            # Удаляем дубликаты:
            paths_to_images = remove_duplicates(paths_to_images, epsilon=8)

            # Открываем изображения и изменяем их размер до 256px по меньшей стороне:
            images = [Image.open(path) for path in paths_to_images]
            images = [resize_image(img, size=256) for img in images]

            # Сохраняем:
            output_dir = os.path.join(dataset_dir, dir_name)
            save_images(images, dir_path=output_dir)

            print(f"Успешно обработано изображений: [{len(images)} / {len(images_names)}].\n")


if __name__ == "__main__":
    main()
