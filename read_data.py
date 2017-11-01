import csv
import numpy as np
from matplotlib.pyplot import imread


def get_data(path='data/default_set/'):
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
        # Reading image
        source_path = line[0]
        img_path = path + source_path
        filenames.append(img_path)
        img = imread(img_path)
        images.append(img)
        # Reading measurements
        measurement = float(line[3])
        measurements.append(measurement)

    X_train = np.array(images)
    y_train = np.array(measurements)

    return X_train, y_train, filenames
