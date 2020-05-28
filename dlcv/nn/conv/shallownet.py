# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class ShallowNet:
    def build(width, height, depth, classes):
        # Block comment -- main builder method
        # build method is main builder of our convnet, it accepts an input
        # image width (coloumns), height (rows), depth (rgb channels) and
        # number of classes (labels) needed to predict

        # initialize the model as sequential
        model = Sequential()
        # initialize the input image shape as "channels last"
        inputShape = (height, width, depth)
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # Block comment -- start building shallownet

        # define the first (and only) CONV => RELU layer
        # 32 filters (K) of size 3x3 and same padding so input
        # size image = output size image 
        model.add(Conv2D(32, (3, 3), padding="same",
        input_shape=inputShape))
        # After the convolution we apply ReLU activation function
        model.add(Activation("relu"))

        # Block comment -- softmax classifier

        #flattten the multi-dimensional representation
        # into 1-D in order to feed to the dense layer
        model.add(Flatten())
        # Dense layer is created using the same number
        # of nodes as our output class labels
        model.add(Dense(classes))
        # applies a softmax activation function which will
        # give us the class label probabilities for each class.
        model.add(Activation("softmax"))
        # return the constructed network architecture
        return model