import scipy.misc


RESIZE_FACTOR = 0.5
TRIM_UP, TRIM_DOWN = 50, 16


def process_img(original_image):
    image = original_image[TRIM_UP:-TRIM_DOWN]
    image = scipy.misc.imresize(image, RESIZE_FACTOR)
    image = image / 127.5 - 1.
    return image
