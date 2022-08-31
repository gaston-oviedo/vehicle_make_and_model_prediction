import os
import yaml
from yaml.loader import SafeLoader
import tensorflow as tf
from tensorflow import keras
import numpy as np


def validate_config(config):
    """
    Takes as input the experiment configuration as a dict and checks for
    minimum acceptance requirements.

    Parameters
    ----------
    config : dict
        Experiment settings as a Python dict.
    """
    if "seed" not in config:
        raise ValueError("Missing experiment seed")

    if "data" not in config:
        raise ValueError("Missing experiment data")

    if "directory" not in config["data"]:
        raise ValueError("Missing experiment training data")


def load_config(config_file_path):
    """
    Loads experiment settings from a YAML file into a Python dict.
    
    Parameters
    ----------
    config_file_path : str
        Full path to experiment configuration file.
        E.g: `/home/app/src/experiments/exp_001/config.yml`

    Returns
    -------
    config : dict
        Experiment settings as a Python dict.
    """
    with open(config_file_path) as f:
        config = yaml.load(f, Loader=SafeLoader)

    validate_config(config)

    return config


def get_class_names(config):
    """
    It's not always easy to track how Keras maps our dataset classes to
    the model outputs.
    Given an image, the model output will be a 1-D vector with probability
    scores for each class. The challenge is, how to map our class names to
    each score in the output vector.
    We will use this function to provide a class order to Keras and keep
    consistency between training and evaluation.

    Parameters
    ----------
    config : dict
        Experiment settings as Python dict.

    Returns
    -------
    classes : list
        List of classes as string.
        E.g. ['AM General Hummer SUV 2000', 'Buick Verano Sedan 2012',
                'FIAT 500 Abarth 2012', 'Jeep Patriot SUV 2012',
                'Acura Integra Type R 2001', ...]
    """
    return sorted(os.listdir(os.path.join(config["data"]["directory"])))


def walkdir(folder):
    """
    Walk through all the files in a directory and its subfolders.

    Parameters
    ----------
    folder : str
        Path to the folder you want to walk.

    Returns
    -------
        For each file found, yields a tuple having the path to the file
        and the file name.
    """
    for dirpath, _, files in os.walk(folder):
        for filename in files:
            yield (dirpath, filename)


def predict_from_folder(folder, model, input_size, class_names):
    """
    Walk through all the image files in a directory, loads them, applies
    the corresponding pre-processing and sends to the model to get
    predictions.

    This function will also return the true label for each image, to do so,
    the folder must be structured in a way in which images for the same
    category are grouped into a folder with the corresponding class
    name. This is the same data structure as we used for training our model.

    Parameters
    ----------
    folder : str
        Path to the folder you want to process.

    model : keras.Model
        Loaded keras model.

    input_size : tuple
        Keras model input size, we must resize the image to math these
        dimensions.

    class_names : list
        List of classes as string. It allow us to map model output IDs to the
        corresponding class name, e.g. 'Jeep Patriot SUV 2012'.

    Returns
    -------
    predictions, labels : tuple
        It will return two lists:
            - predictions: having the list of predicted labels by the model.
            - labels: is the list of the true labels, we will use them to
                      compare against model predictions.
    """
    path_n_files = walkdir(folder)
    pred_list = []
    label_list = []
    while True:
        try:
            next_img = next(path_n_files)
            dir = next_img[0]
            file_name = next_img[1]
            image = keras.preprocessing.image.load_img(os.path.join(dir, file_name))
            input_arr = keras.preprocessing.image.img_to_array(image)
            input_arr = keras.preprocessing.image.smart_resize(
                input_arr, size=input_size
            )
            input_arr = np.array([input_arr])  # Convert single image to a batch.
            pred = model.predict(input_arr)
            pred_idx = np.argmax(pred)
            pred_list.append(class_names[pred_idx])
            true_label = dir.split("/")[-1]
            label_list.append(true_label)
        except StopIteration:
            break

    predictions = pred_list
    labels = label_list

    return predictions, labels

