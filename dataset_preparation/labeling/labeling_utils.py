import os
import shutil


def make_directory(dir_path: str):
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)
