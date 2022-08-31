from utils.data_aug import create_data_aug_layer
import tensorflow as tf
from tensorflow import keras



def create_model(
    weights: str = "imagenet",
    input_shape: tuple = (224, 224, 3),
    dropout_rate: float = 0.0,
    data_aug_layer: dict = None,
    classes: int = None,
    l1_factor: float = 0.01,
    l2_factor: float = 0.01
):
    """
    Creates and loads the Resnet50 model we will use for our experiments.
    Depending on the `weights` parameter, this function will return one of
    two possible keras models:
        1. weights='imagenet': Returns a model ready for performing finetuning
                               on your custom dataset using imagenet weights
                               as starting point.
        2. weights!='imagenet': Then `weights` must be a valid path to a
                                pre-trained model on our custom dataset.
                                This function will return a model that can
                                be used to get predictions on our custom task.

    Parameters
    ----------
    weights : str
        None (random initialization),
        'imagenet' (pre-training on ImageNet), or the path to the
        weights file to be loaded.

    input_shape	: tuple
        Model input image shape as (height, width, channels).
        Only needed when weights='imagenet'. Otherwise, the trained model
        already has the input shape defined and we shouldn't change it.
        Input image size cannot be no smaller than 32. E.g. (224, 224, 3)
        would be one valid value.

    dropout_rate : float
        Value used for Dropout layer to randomly set input units
        to 0 with a frequency of `dropout_rate` at each step during training
        time, which helps prevent overfitting.
        Only needed when weights='imagenet'.

    data_aug_layer : dict
        Configuration from experiment YAML file used to setup the data
        augmentation process during finetuning.
        Only needed when weights='imagenet'.

    classes : int
        Model output classes.
        Only needed when weights='imagenet'. Otherwise, the trained model
        already has the output classes number defined and we shouldn't change
        it.

    Returns
    -------
    model : keras.Model
        Loaded model either ready for performing finetuning or to start doing
        predictions.
    """
    
    if weights == "imagenet":
        
        input = keras.layers.Input(shape=input_shape, dtype=tf.float32)

        if data_aug_layer != None:
            data_augmentation = create_data_aug_layer(data_aug_layer)
            x = data_augmentation(input)
        else:
            x = input
        
        preprocess_input = keras.applications.resnet50.preprocess_input

        core_model = tf.keras.applications.ResNet50(input_shape=input_shape,
                                                    include_top=False, #not including the classifier
                                                    weights=weights,
                                                    pooling='avg'
                                                   )

        drop_out = keras.layers.Dropout(dropout_rate)

        outputs = keras.layers.Dense(classes, kernel_regularizer=keras.regularizers.L1L2(l1=l1_factor, l2=l2_factor)  , activation='softmax')

        x = preprocess_input(x)
        x = core_model(x)
        x = drop_out(x)
        outputs = outputs(x)
        
        model = keras.Model(input, outputs)
    else:
        # For this particular case we want to load our already defined and
        # finetuned model
        model = keras.models.load_model(weights)

    return model
