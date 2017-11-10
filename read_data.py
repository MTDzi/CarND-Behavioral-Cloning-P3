import os
import csv
import numpy as np
from matplotlib.pyplot import imread


def get_data(path='data/default_set/', margin=.25, preprocessing=None):
    lines = []
    with open(path + 'driving_log.csv') as file_:
        reader = csv.reader(file_)
        header = next(reader)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    filenames = []
    for line in lines[:100]:
        # Reading images
        img_paths = list(map(
            lambda sub_path: os.path.join(path, str.strip(sub_path)),
            line[:3]
        ))
        #filenames.extend(img_paths)
        for img_path in img_paths:
            img = imread(img_path)
            if preprocessing:
                img = preprocessing(img)
            images.append(img)

        # Reading measurements
        center_angle = float(line[3])
        angles = [center_angle, center_angle+margin, center_angle-margin]
        measurements.extend(angles)

    X_train = np.array(images)
    y_train = np.array(measurements)

    return X_train, y_train, filenames


if __name__ == '__main__':
    X, y, filenames = get_data(path='mini_data/default_set/')
