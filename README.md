# Simple Convolutional Neural Network (ShallowNet) using Keras on Animal Dataset

------



## Introduction

This project shows up the power of convolutional neural network in computer vision. This network comprises just one convolutional layer, that's why it's named ShallowNet, and achieves good results on classifying three classes on the animal dataset.



## Pre-requisites

- Python 3.6.9
- TensorFlow 1.14
- Keras 2.0.8
- OpenCV 4.2.0
- imutils
- argparse



## The Network

The entire network is as simple as the following:

**INPUT => CONV => RELU => FC**

Thats it! It accepts an input image, apply the convolutions followed by the activation function and the classification is simply done upon that. All implementation are built using Keras with TensorFlow back-end. It shows just a brief introduction on how to build a simple CNN using such great framework.



## The Dataset

The “Animals” dataset is a simple example dataset to demonstrate how to train image classifiers using simple machine learning techniques as well as advanced deep learning algorithms. Images inside the Animals dataset belong to three distinct classes: dogs, cats, and pandas, with 1,000 example images per class.

![](https://github.com/FaroukZidane/ShallowNet-Animal-Classifier/raw/master/doc/images/dataset.png)

Dataset kaggle link: [HERE](https://www.kaggle.com/ashishsaxena2209/animal-image-datasetdog-cat-and-panda/data)



## Training

Download the dataset from the link above. Then simply, execute the following terminal command in the project repository.

```
$ python shallownet_animals.py --dataset PATH_TO_DIR/datasets/animals
```

*Remove `PATH_TO_DIR` and add your own path of the animal dataset folder.*

Training is held at `batch_size =3 2`, `epoch = 100` and `tensorflow_gpu`

## ShallowNet Results

![](https://github.com/FaroukZidane/ShallowNet-Animal-Classifier/raw/master/doc/images/res.png)