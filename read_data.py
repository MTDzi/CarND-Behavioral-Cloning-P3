import os
import csv
import numpy as np
from matplotlib.pyplot import imread


def get_data(path='data/default_set/', margin=2.5):
    lines = []
    with open(path + 'driving_log.csv') as file_:
        reader = csv.reader(file_)
        header = next(reader)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    filenames = []
    for line in lines:
        # Reading images
        img_paths = list(map(
            lambda sub_path: os.path.join(path, str.strip(sub_path)),
            line[:3]
        ))
        filenames.extend(img_paths)
        # import ipdb; ipdb.set_trace()
        for img_path in img_paths:
            try:
                img = imread(img_path)
                images.append(img)
            except FileNotFoundError:
                pass

        # Reading measurements
        center_angle = float(line[3])
        angles = [center_angle, center_angle+margin, center_angle-margin]
        measurements.extend(angles)

    X_train = np.array(images)
    y_train = np.array(measurements)

    return X_train, y_train, filenames


if __name__ == '__main__':
    X, y, filenames = get_data(path='mini_data/default_set/')
