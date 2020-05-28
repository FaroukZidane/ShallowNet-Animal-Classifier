# import the necessary packages
import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        # optionally pass in a list of image preprocessors
        # (such as SimpleProcessor) that can be sequentially
        # applied to a given input image.

        # if the preprocessors are None, initialize them as an
        # empty list, the list makes it easy to apply sequential
        # preprocessing operations on an image rather than
        # applying every operation independently
        if self.preprocessors is None:
            self.preprocessors = []
        
    def load(self, imagePaths, verbose=-1):
        # Our load method requires a single parameter – imagePaths
        # which is a list specifying the file paths to the images
        # in our dataset residing on disk

        # initialize the list of features (images) and class labels
        data = []
        labels = []

        for (i, imagePath) in enumerate(imagePaths):
            # we start looping over each of the input images
            # load the image and extract the class label assuming
            # that our path has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # check to see if our preprocessors are not None
            # as if it is not None, we would loop over each of
            # the preprocessors
            if self.preprocessors is not None:
                # loop over the preprocessors and sequentially
                # apply each one to the image
                for p in self.preprocessors:
                    image = p.preprocess(image)
            
            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)

            # last code block simply handles printing updates
            # to our console and then returning a 2-tuple of
            # the data and labels to the calling function

            # show an update every ‘verbose‘ images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1,
                len(imagePaths)))

        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))