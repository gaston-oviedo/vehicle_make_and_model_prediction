"""
This script will be used to remove noisy background from cars images to
improve the quality of our data and get a better model.
The main idea is to use a vehicle detector to extract the car
from the picture, getting rid of all the background, which may cause
confusion to our CNN model.
We must create a new folder to store this new dataset, following exactly the
same directory structure with its subfolders but with new images.
"""

import argparse
from importlib.resources import path
from utils import utils
from utils.detection import get_vehicle_coordinates
import cv2
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. Already "
            "splitted in train/test sets. E.g. "
            "`/home/app/src/data/car_ims_v1/`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "cropped pictures. E.g. `/home/app/src/data/car_ims_v2/`."
        ),
    )

    args = parser.parse_args()

    return args


def main(data_folder, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to train/test images folder.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        cropped images.
    """    
    path_n_files = utils.walkdir(data_folder)
    while True:
        try:
            next_img = next(path_n_files)
            dir = next_img[0]
            file_name = next_img[1]
            im = cv2.imread(os.path.join(dir, file_name))

            left, top, right, bottom = get_vehicle_coordinates(im)

            cropped_im = im[top:bottom, left:right]

            dir_to = os.path.join(
                output_data_folder, dir.split("/")[-2], dir.split("/")[-1]
            )

            # Create dir if not exists
            if not os.path.exists(dir_to):
                os.mkdir(dir_to)
            path_to = os.path.join(dir_to, file_name)
            # Create img if not exists
            if not os.path.isfile(path_to):
                cv2.imwrite(path_to, cropped_im)
                
        except StopIteration:
            break


if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.output_data_folder)
