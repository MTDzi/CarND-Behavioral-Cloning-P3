import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input, Lambda, Conv2D, Cropping2D, BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras.regularizers import l2

from image_preprocessing import RESIZE_FACTOR, TRIM_DOWN, TRIM_UP


PROCESSED_SHAPE = (
        int(RESIZE_FACTOR*(160-TRIM_UP-TRIM_DOWN)),
        int(RESIZE_FACTOR*320),
        3
) 


def get_lenet_on_steroids_model(l2_reg=1e-3, filter_sz=5, num_filters=16):
    inp_shape = PROCESSED_SHAPE 
    inp = Input(inp_shape)
    inp_ = Conv2D(4, (1, 1), kernel_regularizer=l2(l2_reg))(inp)
    x = Conv2D(num_filters, (filter_sz, filter_sz), kernel_regularizer=l2(l2_reg))(inp_)
    x = MaxPooling2D(2, 2)(x)
    x = ELU()(x)
    x = Conv2D(2*num_filters, (filter_sz, filter_sz), kernel_regularizer=l2(l2_reg))(x)
    x = MaxPooling2D(2, 2)(x)
    x = ELU()(x)
    x = Conv2D(4*num_filters, (filter_sz, filter_sz), kernel_regularizer=l2(l2_reg))(x)
    x = MaxPooling2D(2, 2)(x)
    x = ELU()(x)
    x = Flatten()(x)

    filter_sz += 2
    z = Conv2D(num_filters, (filter_sz, filter_sz), kernel_regularizer=l2(l2_reg))(inp_)
    z = MaxPooling2D(2, 2)(z)
    z = ELU()(z)
    z = Conv2D(2*num_filters, (filter_sz, filter_sz), kernel_regularizer=l2(l2_reg))(z)
    z = MaxPooling2D(2, 2)(z)
    z = ELU()(z)
    z = Conv2D(4*num_filters, (filter_sz, filter_sz), kernel_regularizer=l2(l2_reg))(z)
    z = MaxPooling2D(2, 2)(z)
    z = ELU()(z)
    z = Flatten()(z)

    x = keras.layers.concatenate([x, z])

    print('Shape before Flatten: {}'.format(x))
    x = Dropout(.5)(x)
    x = Dense(512, kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(.5)(x)
    x = ELU()(x)
    x = Dense(256, kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(.5)(x)
    x = ELU()(x)
    x = Dense(128, kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(.5)(x)
    x = ELU()(x)
    x = Dense(64, kernel_regularizer=l2(l2_reg))(x)
    x = ELU()(x)
    x = Dense(1)(x)

    return Model(inp, x)


def get_lenet_like_model(l2_reg=1e-3, filter_sz=5, num_filters=16, use_conv_1x1=True):
    inp_shape = PROCESSED_SHAPE
    inp = Input(inp_shape)

    if use_conv_1x1:
        x = Conv2D(4, (1, 1), kernel_regularizer=l2(l2_reg))(inp)
    else:
        x = inp

    x = Conv2D(num_filters, (filter_sz, filter_sz),
               padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = MaxPooling2D(2, 2)(x)
    x = ELU()(x)

    x = Conv2D(2*num_filters, (filter_sz, filter_sz),
               padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = MaxPooling2D(2, 2)(x)
    x = ELU()(x)

    x = Conv2D(4*num_filters, (filter_sz, filter_sz),
               padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = MaxPooling2D(2, 2)(x)
    x = ELU()(x)

    print('Shape before Flatten: {}'.format(x))
    x = Dropout(.5)(x)
    x = Flatten()(x)
    x = Dense(256, kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(.5)(x)
    x = ELU()(x)
    x = Dense(128, kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(.5)(x)
    x = ELU()(x)
    x = Dense(64, kernel_regularizer=l2(l2_reg))(x)
    x = ELU()(x)
    x = Dense(1, kernel_regularizer=l2(l2_reg))(x)

    return Model(inp, x)


def get_nvidia_model(l2_reg=1e-3, filter_sz=5, num_filters=16):
    inp_shape = PROCESSED_SHAPE
    inp = Input(inp_shape)

    x = Conv2D(24, (5,5), strides=(2,2), padding='same', kernel_regularizer=l2(l2_reg))(inp)
    x = ELU()(x)
    x = Conv2D(36, (5,5), strides=(2,2), padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = ELU()(x)
    x = Conv2D(48, (5,5), strides=(2,2), padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = ELU()(x)

    x = Conv2D(64, (3,3), kernel_regularizer=l2(l2_reg))(x)
    x = ELU()(x)
    x = Conv2D(64, (3,3), kernel_regularizer=l2(l2_reg))(x)
    x = ELU()(x)

    x = Flatten()(x)
    x = Dense(100, kernel_regularizer=l2(l2_reg))(x)
    x = ELU()(x)
    x = Dense(50, kernel_regularizer=l2(l2_reg))(x)
    x = ELU()(x)
    x = Dense(10, kernel_regularizer=l2(l2_reg))(x)
    x = ELU()(x)
    x = Dense(1)(x)

    return Model(inp, x)



if __name__ == '__main__':
    from read_data import get_data_gen, batcher
    from image_preprocessing import process_img
    data_filepath = 'data/new_data/'

    # Surprisingly, my best model was the "LeNet like model" (my own design)
    model = get_lenet_like_model()
    model.compile(loss='mse',
                  optimizer=Adam(lr=1e-4),
                  metrics=['mse'])

    batch_sz = 32
    epoch_sz = 60000

    train_gen = get_data_gen(
        data_filepath,
        flip_prob=0.25
    )
    train_batch_gen = batcher(train_gen, batch_sz)

    valid_gen = get_data_gen(
        data_filepath,
        validation=True,
        flip_prob=0.0
    )
    valid_batch_gen = batcher(valid_gen, batch_sz)

    # Initialize generators
    _, _ = next(train_batch_gen)
    _, _ = next(valid_batch_gen)

    model.fit_generator(
            train_batch_gen,
            steps_per_epoch=epoch_sz//batch_sz,
            epochs=10,
            use_multiprocessing=True,
            validation_data=valid_batch_gen,
            validation_steps=300,
    )
    model.save('model_with_conv_1x1_10.h5') 

    model.fit_generator(
            train_batch_gen,
            steps_per_epoch=epoch_sz//batch_sz,
            epochs=1,
            use_multiprocessing=True,
            validation_data=valid_batch_gen,
            validation_steps=300,
    )
    model.save('model_with_conv_1x1_11.h5') 

    model.fit_generator(
            train_batch_gen,
            steps_per_epoch=epoch_sz//batch_sz,
            epochs=1,
            use_multiprocessing=True,
            validation_data=valid_batch_gen,
            validation_steps=300,
    )
    model.save('model_with_conv_1x1_12.h5') 
