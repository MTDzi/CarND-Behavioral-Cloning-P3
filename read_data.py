import os
import csv
import numpy as np
import pandas as pd
from matplotlib.pyplot import imread



def get_data_gen(path='data/default_set/', margin=.1, preprocessing=None, flip_prob=.5, val_part=100, validation=False, drop_angle_0_prob=0):
    lines = []
    with open(path + 'driving_log.csv') as file_:
        reader = csv.reader(file_)
        header = next(reader)
        for line in reader:
            lines.append(line)

    import ipdb; ipdb.set_trace()
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
                index = np.random.choice([0,1,2], p=[.5,.25,.25])

            center_angle = float(line[3])
            if not validation and index == 0 and center_angle == 0 and np.random.rand() < drop_angle_0_prob:
                continue
            angles = [center_angle, center_angle+margin, center_angle-margin]
            angle = angles[index]

            image = imread(os.path.join(path, str.strip(line[index])))
            if preprocessing:
                image = preprocessing(image)

            if np.random.rand() < flip_prob:
                image, angle = np.fliplr(image), -angle
            yield image, angle


if __name__ == '__main__':
    X, y, filenames = get_data_gen(path='data/default_set/')
