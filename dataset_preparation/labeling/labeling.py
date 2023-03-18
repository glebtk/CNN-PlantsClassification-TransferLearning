import os
import random
import shutil
import pandas as pd

from labeling_utils import make_directory


def labeling(dataset_dir: str, test_number: int = 30, remove_source: bool = False):
    # Renaming the dataset directory:
    source_dir = "./data/dataset_source"
    os.rename(dataset_dir, source_dir)

    # Make new directories to save the result:
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    make_directory(dataset_dir)  # General dataset directory
    make_directory(train_dir)  # Train images
    make_directory(test_dir)  # Test images

    train_labels = pd.DataFrame(columns=["path", "label"])
    test_labels = pd.DataFrame(columns=["path", "label"])

    dirs = os.listdir(source_dir)

    # Cycle through all directories with images:
    for label, dir_ in enumerate(dirs):
        current_dir_path = os.path.join(source_dir, dir_)

        if not os.path.isdir(current_dir_path):
            continue

        print(f"\033[32m\033[1mProcessing:\033[0m {label+1}. {dir_}")

        # Create the corresponding directories in train and test:
        make_directory(os.path.join(train_dir, dir_))
        make_directory(os.path.join(test_dir, dir_))

        label_name = dir_.lower().replace(" ", "_").replace("-", "_")

        image_names = os.listdir(current_dir_path)  # Names of images in the current directory
        random.shuffle(image_names)  # Shuffle the names of the images

        # Split into test and train:
        train_number = len(image_names) - test_number
        train_image_names = image_names[:train_number]
        test_image_names = image_names[train_number:]

        # Cycle through the training sample:
        for number, name in enumerate(train_image_names):
            new_name = label_name + "_train_" + str(number + 1) + ".jpg"  # New image name

            old_path = os.path.join(current_dir_path, name)  # Current path to the image
            new_path = os.path.join(*[train_dir, dir_, new_name])  # New path to the image

            # Adding to the csv file:
            record = pd.DataFrame({"path": [f"train/{dir_}/{new_name}"], "label": [label]})
            train_labels = pd.concat([train_labels, record])

            # Moving and renaming the image:
            shutil.copyfile(old_path, new_path)

        # Cycle through the test sample:
        for number, name in enumerate(test_image_names):
            new_name = label_name + "_test_" + str(number + 1) + ".jpg"  # New image name

            old_path = os.path.join(current_dir_path, name)  # Current image path
            new_path = os.path.join(*[test_dir, dir_, new_name])  # New image path

            # Adding to the csv file:
            record = pd.DataFrame({"path": [f"test/{dir_}/{new_name}"], "label": [label]})
            test_labels = pd.concat([test_labels, record])

            # Moving and renaming the image:
            shutil.copyfile(old_path, new_path)

    # If necessary, delete the original directory:
    if remove_source:
        shutil.rmtree(source_dir)

    # Saving csv-fies:
    train_labels.to_csv("./data/dataset/train_labels.csv", index=False)
    test_labels.to_csv("./data/dataset/test_labels.csv", index=False)

    print("\033[32m\033[1m\n\t...Done!\033[0m")


if __name__ == "__main__":
    labeling("./data/dataset/", remove_source=False)
