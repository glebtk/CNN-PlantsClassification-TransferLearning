import os
import shutil
import imagehash

from PIL import Image
from PIL import ImageFile


def make_directory(dir_path: str):
    """
    Description
    ---------
    Creates a new directory. If the directory already exists, overwrites it.

        Parameters
        ---------
        **dir_path:** str
            The path where the directory will be created
    """

    try:
        os.makedirs(dir_path)
    except FileExistsError:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)


def verify_image(path: str) -> bool:
    """
    Description
    ---------
    Checks if the file is a valid image.

    Parameters
    ---------
    **path:** string
        Image path

    Return value
    ---------
    bool: A binary value describing whether the file is an image
    """

    try:
        Image.open(path)
        return True
    except Exception:
        return False


def check_filters(path: str, min_size: int = 1, mode: str = "RGB") -> bool:
    """
    Description
    ---------
    Checks an image for compliance with filters.

    Args
    ---------
    **path:** str
        Path to the image
    **min_size:** int
        Minimum image size on the smaller side
    **mode:** str
        Color mode

    Return value
    ---------
    bool: The value of the image's compliance (or non-compliance) with filters
    """

    image = Image.open(path)
    if min(image.size) < min_size or image.mode != mode:
        return False
    else:
        return True


def remove_duplicates(paths: list, epsilon: int = 1) -> list:
    """
    Description
    ---------
    Removes paths to identical or similar images from the list.

    Args
    ---------
    **paths:** list
        List of paths to images
    **epsilon:** int
        The similarity between the images

    Return value
    ---------
    list: List of paths to unique images
    """

    ImageFile.LOAD_TRUNCATED_IMAGES = True  # For process damaged images

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
    Description
    ---------
    Changes the size of the image.

    Args
    ---------
    **image:** Image
        PIL-image.
    **size:** int
        Target image size on the smaller side (px)

    Return value
    ---------
    Image: Resized image
    """

    factor = size / min(image.size)
    new_size = (int(image.width * factor), int(image.height * factor))
    return image.resize(new_size)


def resize_images(images: list, size: int) -> list:
    """
    Description
    ---------
    Changes the size of the images in the list.

    Args
    ---------
    **images:** list
        List of PIL-images.
    **size:** int
        Target image size on the smaller side (px)

    Return value
    ---------
    list: List of resized images
    """
    resized_images = []
    for image in images:
        factor = size / min(image.size)
        new_size = (int(image.width * factor), int(image.height * factor))
        resized_images.append(image.resize(new_size))

    return resized_images


def save_images(images: list, dir_path: str):
    """
    Description
    ---------
    Creates a new directory at the specified path, and saves images to it.

    Args
    ---------
    **images:** list
        List of PIL-images.
    **dir_path:** str
        Path to directory
    """
    make_directory(dir_path)
    for i, image in enumerate(images):
        img_name = f"{i}.jpg"
        img_path = os.path.join(dir_path, img_name)
        image.save(img_path)
