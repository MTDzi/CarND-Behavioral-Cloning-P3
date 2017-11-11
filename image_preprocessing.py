import numpy as np
import cv2
import matplotlib.image as mpimg
from PIL import Image
import scipy


KERNEL_SIZE = 5
RESIZE_FACTOR = .5
VERTICES = np.array([[
    (0, 60),
    (80, 40),
    (240, 40),
    (320, 60),
    (320, 160),
    (0, 160),
]], dtype=np.int32)


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_img(original_image):
    image = scipy.misc.imresize(original_image, RESIZE_FACTOR) 
    ''' 
    image = original_image / 255. - 0.5
    image = cv2.resize(
            image,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
    )
    '''
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #blur_gray = cv2.GaussianBlur(image, (KERNEL_SIZE, KERNEL_SIZE), 0)
    edges = cv2.Canny(image, 10, 200)
    masked_edges = roi(edges, [VERTICES])

    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 40     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = int(RESIZE_FACTOR*20) #minimum number of pixels making up a line
    max_line_gap = 2    # maximum gap in pixels between connectable line segments
    line_image = np.copy(gray_image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    lines = [] if lines is None else lines
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image, (x1,y1), (x2,y2), (255,255,255), 2)

    processed_image = line_image + edges
    processed_image = processed_image / 255.
    gray_image = gray_image / 255. - 0.5
    return np.concatenate([
        np.expand_dims(processed_image, 2),
        np.expand_dims(gray_image, 2),
        image,
    ], 2)
