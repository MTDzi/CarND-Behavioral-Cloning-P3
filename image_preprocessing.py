import numpy as np
import cv2
import matplotlib.image as mpimg
from PIL import Image


KERNEL_SIZE = 5
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
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray_image, (KERNEL_SIZE, KERNEL_SIZE), 0)
    edges = cv2.Canny(gray_image, 200, 300)
    masked_edges = roi(edges, [VERTICES])

    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 40     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20 #minimum number of pixels making up a line
    max_line_gap = 2    # maximum gap in pixels between connectable line segments
    line_image = np.copy(gray_image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image, (x1,y1), (x2,y2), (255,255,255), 2)

    processed_image = 255 - (line_image + edges)
    return np.concatenate([
        np.expand_dims(processed_image, 2),
        np.expand_dims(gray_image, 2)
    ], 2)