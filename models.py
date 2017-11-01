import numpy as np

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input, Lambda, Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping


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


def get_lenet_model():
    inp = Input(shape=(160, 320, 3))
    x = Cropping2D(cropping=((70, 25), (0,0)))
    x = Lambda(lambda x: (x / 255.0) - 0.5)(inp)
    x = Conv2D(10, (5, 5), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(20, (5, 5), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(40, (5, 5), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)

    print('Shape before Flatten: {}'.format(x.get_shape()))
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Dropout(.5)(x)
    x = Dense(128)(x)
    x = Dropout(.5)(x)
    x = Dense(1)(x)

    return Model(inp, x)


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


if __name__ == '__main__':
    from read_data import get_data
    model = get_lenet_model()
    X, y, filenames = get_data('data/default_set/')
    
    X_flip, y_flip = np.fliplr(X), -y

    model.compile(loss=root_mean_squared_error,
                  optimizer='adam',
                  metrics=[root_mean_squared_error])
    model.fit(X, y, epochs=10, verbose=2, validation_data=(X_flip, y_flip))
