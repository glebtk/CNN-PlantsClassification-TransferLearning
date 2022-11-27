import os
import torch
import config
import pandas as pd

from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.nn.functional import one_hot


class CrimeanPlantsDataset(Dataset):
    def __init__(self, root_dir: str, csv_file: str, oversampling: bool = False, transform=None,):
        self.root_dir = root_dir

        if oversampling:
            df_csv = pd.read_csv(csv_file)
            max_class_length = df_csv["label"].value_counts().max()
            class_labels = df_csv["label"].unique()

            data_csv = pd.DataFrame()
            for label in class_labels:
                current_class = df_csv.loc[df_csv["label"] == label]

                add_samples_number = max_class_length - len(current_class)

                current_class_oversampled = pd.concat(
                    [
                        current_class,
                        current_class.sample(add_samples_number, replace=True)
                    ]
                )

                data_csv = pd.concat([data_csv, current_class_oversampled])

            self.data_csv = data_csv
        else:
            self.data_csv = pd.read_csv(csv_file)

        self.transform = transform

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.data_csv.iloc[index, 0])

        img = read_image(img_path).type(torch.FloatTensor)
        img /= 255

        if self.transform:
            img = self.transform(img)

        label = one_hot(torch.tensor(self.data_csv.iloc[index, 1]), config.OUT_FEATURES).type(torch.FloatTensor)

        return img, label


def dataset_test():
    root_dir = "./data/dataset/"
    csv_file = "./data/dataset/train_labels.csv"
    dataset = CrimeanPlantsDataset(root_dir=root_dir, csv_file=csv_file, oversampling=True)


if __name__ == "__main__":
    dataset_test()
