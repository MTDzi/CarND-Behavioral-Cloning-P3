import os
import csv
import numpy as np
import pandas as pd
from matplotlib.pyplot import imread

from image_preprocessing import process_img as preprocessing


def get_data_gen(path='data/default_set/', margin=.25, flip_prob=.5, val_part=100, validation=False):
    lines = []
    with open(path + 'driving_log.csv') as file_:
        reader = csv.reader(file_)
        header = next(reader)
        for line in reader:
            lines.append(line)

    if not validation:
        print('Num lines: {}'.format(len(lines)))

    np.random.seed(42)
    while True:
        np.random.shuffle(lines)
        for line in lines:
            is_for_val = (hash(line[0]) % val_part != 0)
            if is_for_val:
                if not validation:
                    continue
            else:
                if validation:
                    continue

            if validation:
                index = 0
            else:
                p = .25
                index = np.random.choice([0,1,2], p=[1-2*p,p,p])

            center_angle = float(line[3])
            if not validation and np.random.rand() > np.abs(center_angle):
                continue

            angles = [center_angle, center_angle+margin, center_angle-margin]
            angle = angles[index]

            image = imread(os.path.join(path, str.strip(line[index])))
            image = preprocessing(image)

            if np.random.rand() < flip_prob:
                image, angle = np.fliplr(image), -angle

            yield image, angle


def batcher(gen, batch_size):
    while True:
        X = []
        Y = []
        for i in range(batch_size):
            x, y = next(gen)
            X.append(x)
            Y.append(y)
        yield np.array(X), np.array(Y)
