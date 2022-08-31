# Resnet 50 Training

## Overview

In this project we train and evaluate a convolutional neuron network (Resnet50).

To train and evaluate the model we use a data set of cars and trucks. Data can be downloaded from [here](http://ai.stanford.edu/~jkrause/cars/car_dataset.html). If you have the data in a amazon bucket (as I did), scripts/get_dataset.py script can help with that. 

After an initial aproximation, we cropped the images using a detector (detectron2) already trained, in order to only have the car or truck in each image. 

To train and evaluate I used AWS servers. 

All the libraries and dependecies are taken care of in a docker container. 

## Baseline

The baseline for this projects is the results of training and validation in a general configuration. On top of that results, we iterate in order to improve the metrics. 

## Non Cropped Images

In this stage of training the model is fed and tested with the images without crop. 

###    Training

During training a sweep of hyperparameter was made. Among the values that were tested are: 

*batch_size: 16, 32.  
dropout_rate (in the dense layer): 0, 0.1, 0.2, 0.3, 0.4, 0.5  
data_augmentation: random_flip (horizontal, vertical and combined), random_rotation: [-0.025, 0.025; 0.1, 0.2], random_zoom (0.1, 0.2). No data augmentation was also tested.  
learning_rate: 0.001, 0,01, 0.025  
epsilon: 0.0001, 0.1  
Regularization L1 y L2: disabled and 0.01 as a factor*

Thanks to the various regularization method (drop out, data augmentation, L1, L2) validation performance improved a lot respect to training performance. 


###    Evaluation

With the best model so far, the model was evaluated. The overall accuracy obtained was: **0.64**.

## Cropped Images

In this stage of training the model is fed and tested with removed background images. For this, a trained detectron2 was used.

### Training

For training the model with the cropped images, the same hyperparameters that were used in the best model obtained with the non cropped images were used. 

The overall performance was improved in training and validation by removing the images background

### Evaluation

The previous model was evaluated in the test dataset. The overall accuracy obtained was: **0.82**.


## Conclusion

In all test that were performed, an overfit was observed. The main reason is that there are just a few images per class. So the model tends to memorize the train dataset. With regularizacion techniques we improved a lot the validation performance. Without background in images, training and validation performance improved even more. 

On the evaluation side, it was ovserved that the performance was a little worse than the validation one. This could be because model may overfit also the validation data as we tried to improve the performance in this dataset. 

# Development

## 1. Install

You can use `Docker` to easily install all the needed packages and libraries. Two Dockerfiles are provided for both CPU and GPU support.

- **CPU:**

```bash
$ docker build -t s05_project -f docker/Dockerfile .
```

- **GPU:**

```bash
$ docker build -t s05_project -f docker/Dockerfile_gpu .
```

### Run Docker CPU

```bash
    $ docker run --rm --net host -it \
        -v $(pwd):/home/app/src \
        --workdir /home/app/src \
        s05_project \
        bash
```
### Run Docker GPU
```bash
$ docker run --rm --net host --gpus all -it \
    -v $(pwd):/home/app/src \
    --workdir /home/app/src \
    s05_project \
    bash
```

### Run Unit test

```bash
$ pytest tests/
```

## 2. Prepare the data

After download and process the data, it should look like this:

```
data/
    ├── car_dataset_labels.csv
    ├── car_ims
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   ├── ...
```

Then, you should be able to run the script `scripts/prepare_train_test_dataset.py`. It will format the data in a way Keras can use for training our CNN model.

An EDA can be seen in notebooks folder.


## 3. Training

After we have the images in place, it's time to create a CNN and train it on the dataset. To do so, we will make use of `scripts/train.py`.

The only input argument it receives is a YAML file with all the experiment settings like dataset, model output folder, epochs,
learning rate, data augmentation, etc.

Each time you are going to train a new a model, you should create a new folder inside the `experiments/` folder with the experiment name. Inside this new folder, create a `config.yml` with the experiment settings. Weights and training logs should be stored inside the same experiment folder to avoid mixing things between different runs. The folder structure should look like this:

```bash
experiments/
    ├── exp_001
    │   ├── config.yml
    │   ├── logs
    │   ├── model.01-6.1625.h5
    │   ├── model.02-4.0577.h5
    │   ├── model.03-2.2476.h5
    │   ├── model.05-2.1945.h5
    │   └── model.06-2.0449.h5
    ├── exp_002
    │   ├── config.yml
    │   ├── logs
    │   ├── model.01-7.4214.h5
    ...
```

The script `scripts/train.py` makes use of external functions from other project modules:

- `utils.load_config()`: Takes as input the path to an experiment YAML configuration file, loads it and returns a dict.
- `resnet50.create_model()`: Returns a CNN ready for training or for evaluation, depending on the input parameters received. 
- `data_aug.create_data_aug_layer()`: Used by `resnet50.create_model()`. This function adds data augmentation layers to our model that will be used only while training.

## 4. Evaluate your trained model

After running many experiments and having a potentially good model trained. It's time to check its performance on our test dataset.

We will use `utils.predict_from_folder()` function for that.

## 5. Improve classification by removing noisy background

As we already saw in the `notebooks/EDA.ipynb` file. Most of the images have a of background which may affect our model learning during the training process.

It's a good idea to remove this background. One thing we can do is to use a Vehicle detector to isolate the car from the rest of the content in the picture.

We use [Detectron2](https://github.com/facebookresearch/detectron2) framework for this. It offers a lot of different models, you can check in its [Model ZOO](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#faster-r-cnn). We use the model called "R101-FPN".

In particular, we use a detector model trained on [COCO](https://cocodataset.org) dataset which has a good balance between accuracy and speed. This model can detect up to 80 different types of objects but here we're only interested on getting two out of those 80, those are the classes "car" and "truck".

- `scripts/remove_background.py`: It processes the initial dataset used for training the model on **item (3)**, removing the background from pictures and storing the resulting images on a new folder.
- `utils/detection.py`: This module loads our detector and implements the logic to get the vehicle coordinate from the image.

Now you have the new dataset in place, it's time to start training a new model and checking the results in the same way as we did for steps items **(3)** and **(4)**.