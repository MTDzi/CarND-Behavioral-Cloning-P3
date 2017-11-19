import numpy as np
import cv2
import matplotlib.image as mpimg
from PIL import Image
import scipy.misc

RESIZE_FACTOR = 0.5


def process_img(original_image):
    image = original_image[50:]
    image = scipy.misc.imresize(image, RESIZE_FACTOR) 
    image = image / 127.5 - 1.
    return image
