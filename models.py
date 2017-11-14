import numpy as np
import cv2

from keras import backend as K
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input, Lambda, Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras.regularizers import l2

from image_preprocessing import RESIZE_FACTOR


def get_simplest_model():
    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))
    return model


def get_application_model(model_name='vgg19', input_tensor=None, img_sizes=(160, 320), include_top=False):
    if include_top:
        assert img_sizes == (224, 224)

    ApplicationModel = {
        'vgg19': applications.VGG19,
        'xception': applications.Xception,
        'mobile_net': applications.MobileNet,
    }[model_name]

    model = ApplicationModel(
        weights = "imagenet",
        include_top=include_top,
        # input_shape=img_sizes+(3,),
        input_tensor=input_tensor,
    )
    for layer in model.layers[:1]:
        layer.trainable = False

    return model


def resize_normalize(image):
    import tensorflow as tf
    image = image/255.0 - 0.5
    resized = tf.image.resize_images(image, (224, 224))
    return resized


def get_model_with_standardization(model_name='vgg19', include_top=True):
    in_shape = (160, 320, 3)
    input_ = Input(shape=in_shape)
    x = Lambda(resize_normalize, input_shape=in_shape, output_shape=(224, 224, 3))(input_)
    model = get_application_model(
        model_name=model_name,
        input_tensor=x,
        img_sizes=(224, 224),
        include_top=include_top
    )
    return model


def get_model_extended(model_name='vgg19', include_top=True):
    model = get_model_with_standardization(include_top=include_top)
    output = model.output

    bottleneck_shape = output.get_shape()
    print('Bottleneck shape: {}'.format(bottleneck_shape))
    if len(bottleneck_shape) > 2:
        output = Flatten()(output)

    output = Dense(256, activation='relu')(output)
    output = Dropout(.5)(output)
    output = Dense(1)(output)

    new_model = Model(model.input, output)
    return new_model


def preprocess_img(original_image):
    original_image = (original_image / 255.0) - 0.5
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

    return line_image + edges

def get_lenet_model():
    reg = 1e-3
    filter_sz = 5
    num_filters = 16
    padding='valid'

    inp_shape = (int(RESIZE_FACTOR*110), int(RESIZE_FACTOR*320), 2)
    inp = Input(inp_shape)
    x = Conv2D(num_filters, (filter_sz, filter_sz), kernel_regularizer=l2(reg))(inp)
    x = ELU()(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(2*num_filters, (filter_sz, filter_sz), kernel_regularizer=l2(reg))(x)
    x = ELU()(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(4*num_filters, (filter_sz, filter_sz), kernel_regularizer=l2(reg))(x)
    x = ELU()(x)
    x = MaxPooling2D(2, 2)(x)

    print('Shape before Flatten: {}'.format(x))
    x = Flatten()(x)
    x = Dense(100, kernel_regularizer=l2(reg))(x)
    x = ELU()(x)
    #x = Dropout(.5)(x)
    x = Dense(50, kernel_regularizer=l2(reg))(x)
    x = ELU()(x)
    x = Dense(10, kernel_regularizer=l2(reg))(x)
    x = ELU()(x)
    x = Dense(1)(x)

    return Model(inp, x)


def get_nvidia_model():
    reg = 1e-3
    inp_shape = (int(RESIZE_FACTOR*110), int(RESIZE_FACTOR*320), 3)
    inp = Input(inp_shape)

    x = Conv2D(24, (5,5), strides=(2,2), padding='same', kernel_regularizer=l2(reg))(inp)
    x = ELU()(x)
    x = Conv2D(36, (5,5), strides=(2,2), padding='same', kernel_regularizer=l2(reg))(x)
    x = ELU()(x)
    x = Conv2D(48, (5,5), strides=(2,2), padding='same', kernel_regularizer=l2(reg))(x)
    x = ELU()(x)

    x = Conv2D(64, (3,3), kernel_regularizer=l2(reg))(x)
    x = ELU()(x)
    x = Conv2D(64, (3,3), kernel_regularizer=l2(reg))(x)
    x = ELU()(x)

    x = Flatten()(x)
    x = Dense(100, kernel_regularizer=l2(reg))(x)
    x = ELU()(x)
    x = Dense(50, kernel_regularizer=l2(reg))(x)
    x = ELU()(x)
    x = Dense(10, kernel_regularizer=l2(reg))(x)
    x = ELU()(x)
    x = Dense(1)(x)

    return Model(inp, x)




def batcher(gen, batch_size=32):
    while True:
        X = []
        Y = []
        for i in range(batch_size):
            x, y = next(gen)
            X.append(x)
            Y.append(y)
        yield np.array(X), np.array(Y)



if __name__ == '__main__':
    from read_data import get_data, get_data_gen
    from image_preprocessing import process_img
    data_filepath = 'data/largest_set/'

    #model = get_lenet_model()
    model = get_nvidia_model()
    model.compile(loss='mse',
                  optimizer=Adam(lr=1e-4),
                  metrics=['mse'])
    '''X, y, filenames = get_data(data_filepath, preprocessing=process_img)
    not_junk = np.where(np.abs(y) > 0.1)[0]
    X, y = X[not_junk], y[not_junk]
    print('Data in RAM')
    #X = np.concatenate([X, np.fliplr(X)])
    #y = np.concatenate([y, -y])
    model.fit(X, y, epochs=20, verbose=1, validation_split=0.01)'''
    batch_sz = 32
    epoch_sz = 48000
    train_gen = get_data_gen(data_filepath, preprocessing=process_img, flip_prob=0.25, drop_angle_0_prob=0.5)
    train_batch_gen = batcher(train_gen, batch_size=batch_sz)
    valid_gen = get_data_gen(data_filepath, preprocessing=process_img, validation=True, flip_prob=0.25, drop_angle_0_prob=0)
    valid_batch_gen = batcher(valid_gen, batch_size=batch_sz)
    model.fit_generator(
            train_batch_gen,
            steps_per_epoch=epoch_sz//batch_sz,
            epochs=20,
            use_multiprocessing=True,
            validation_data=valid_batch_gen,
            validation_steps=200,
    )
    model.save('model.h5')
