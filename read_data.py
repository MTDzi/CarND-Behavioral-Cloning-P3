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
    for line in lines:
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



def get_data_gen(path='data/default_set/', margin=.35, preprocessing=None, flip_prob=.5, val_part=100, validation=False, drop_angle_0_prob=0):
    lines = []
    with open(path + 'driving_log.csv') as file_:
        reader = csv.reader(file_)
        header = next(reader)
        for line in reader:
            lines.append(line)

    np.random.seed(42)
    while True:
        np.random.shuffle(lines)
        for line in lines:
            images = []
            is_for_val = (hash(line[0]) % val_part != 0)
            if is_for_val:
                if not validation:
                    continue
            else:
                if validation:
                    continue

            index = np.random.randint(0, 3)
            center_angle = float(line[3])
            if index == 0 and center_angle == 0 and np.random.rand() < drop_angle_0_prob:
                continue

            angles = [center_angle, center_angle+margin, center_angle-margin]
            angle = angles[index]
           
            # Reading one image
            image = imread(os.path.join(path, str.strip(line[index])))
            if preprocessing:
                image = preprocessing(image)

            if np.random.rand() < flip_prob:
                image, angle = np.fliplr(image), -angle
            yield image, angle


if __name__ == '__main__':
    X, y, filenames = get_data(path='mini_data/default_set/')
