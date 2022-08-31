import tensorflow as tf
from tensorflow import keras


def create_data_aug_layer(data_aug_layer):
    """
    Function to parse the data augmentation methods for the
    experiment and create the corresponding layers.

    Parameters
    ----------
    data_aug_layer : dict
        Data augmentation settings coming from the experiment YAML config
        file.

    Returns
    -------
    data_augmentation : keras.Sequential
        Sequential model having the data augmentation layers inside.
    """

    data_aug_layers = []

    if "random_flip" in data_aug_layer:
        data_aug_layers.append(keras.layers.RandomFlip(**data_aug_layer["random_flip"]))

    if "random_rotation" in data_aug_layer:
        data_aug_layers.append(
            keras.layers.RandomRotation(**data_aug_layer["random_rotation"])
        )

    if "random_zoom" in data_aug_layer:
        data_aug_layers.append(keras.layers.RandomZoom(**data_aug_layer["random_zoom"]))

    data_augmentation = keras.Sequential(data_aug_layers)

    return data_augmentation
