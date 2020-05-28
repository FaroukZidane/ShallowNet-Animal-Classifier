import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter
    
    def preprocess(self, image):
        # get an image as an input argument in order to
        # resize it to a fixed size, ignoring the aspect
        # ratio
        # preprocess the image by resizing it to a fixed
        # size of width and height which
        # then return to the calling function.
        return cv2.resize(image, (self.width, self.height),
        interpolation=self.inter)