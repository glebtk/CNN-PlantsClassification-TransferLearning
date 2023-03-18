import os
import torch
import config
import pandas as pd

from torch.utils.data import Dataset
from torchvision.io import read_image


class CrimeanPlantsDataset(Dataset):
    def __init__(self, root_dir: str, csv_file: str, oversampling: bool = False, transform=None):
        self.root_dir = root_dir

        if oversampling:
            df_csv = pd.read_csv(csv_file)

            # Find the number of images in the most complete class:
            max_class_length = df_csv["label"].value_counts().max()

            # Getting all class labels:
            class_labels = df_csv["label"].unique()

            # Equalize the number of images in the classes by random duplication:
            data_csv = pd.DataFrame()
            for label in class_labels:
                current_class = df_csv.loc[df_csv["label"] == label] # All samples of the current class

                add_samples_number = max_class_length - len(current_class)  # Number of samples to add

                # Randomly select the required number of samples and add them to the existing ones:
                # (replace=True means that the selected samples can be repeated)
                current_class_oversampled = pd.concat(
                    [
                        current_class,
                        current_class.sample(add_samples_number, replace=True)
                    ]
                )

                data_csv = pd.concat([data_csv, current_class_oversampled])  # Adding the class to the general list

            self.data_csv = data_csv
        else:
            self.data_csv = pd.read_csv(csv_file)

        self.transform = transform

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.data_csv.iloc[index, 0])  # Full path to the image

        img = read_image(img_path).type(torch.FloatTensor)  # Open the image and convert it to a tensor
        img /= 255  # Standardization

        # If necessary, apply transformations:
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.data_csv.iloc[index, 1])  # Class label. int, tensor.

        return img, label


def dataset_test():
    root_dir = "./data/dataset/"
    csv_file = "./data/dataset/train_labels.csv"
    dataset = CrimeanPlantsDataset(root_dir=root_dir, csv_file=csv_file, oversampling=True)


if __name__ == "__main__":
    dataset_test()
