# import the necessary packages
from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # The constructor accepts an optional parameter named dataFormat.
        # This value defaults to None indicating that the setting inside
        # keras.json should be used.

        # store the image data format
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # apply the Keras utility function that correctly rearranges
        # the dimensions of the image
        # 1. Accepts an image as input.
        # 2. Calls img_to_array on the image, ordering the channels based
        # on our configuration file/the value of dataFormat.
        # 3. Returns a new NumPy array with the channels properly ordered.
        return img_to_array(image, data_format=self.dataFormat)