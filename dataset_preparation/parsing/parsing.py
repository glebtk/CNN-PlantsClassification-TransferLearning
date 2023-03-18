import os
import time
import pandas as pd

from dataset_preparation.parsing.parsing_utils import find_images
from dataset_preparation.parsing.parsing_utils import find_similar_images
from dataset_preparation.parsing.parsing_utils import make_directory
from dataset_preparation.parsing.parsing_utils import remove_duplicates
from dataset_preparation.parsing.parsing_utils import randomize_delay
from dataset_preparation.parsing.parsing_utils import save_images


def main():
    img_dir = "./data/images"
    number_of_images = 200
    number_of_similar_images = 20
    delay = 6.0

    df_plants = pd.read_csv("./data/crimean_plants.csv")

    # Going through the cycle of Latin names of plants:
    for plant in df_plants["Latin"]:
        print(f"\033[32m\033[1mCurrent plant:\033[0m {plant}.")

        # Get links to images with a plant:
        print(f"\n1.Image search by request \"{plant}\":")
        plants_urls = find_images(plant=plant, number=number_of_images, image_type="photo", delay=randomize_delay(delay))

        # We get links to images similar to the first number_of_similar_images:
        print(f"\n2.Search for images similar to the first {number_of_similar_images} found:")
        for url in plants_urls[:number_of_similar_images]:
            time.sleep(randomize_delay(delay))
            similar_plants_urls = find_similar_images(url=url, number=40, delay=randomize_delay(delay))
            plants_urls += similar_plants_urls

        # Removing duplicate links:
        plants_urls = remove_duplicates(plants_urls)

        # Save the found images:
        print("\n3.Uploading:")

        dir_path = os.path.join(img_dir, str(plant))
        make_directory(dir_path=dir_path)

        save_images(urls=plants_urls, dir_path=dir_path, number_images=True)


if __name__ == "__main__":
    main()
