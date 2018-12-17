from keras.layers import *
from keras.models import Model
from keras.optimizers import *

from keras import layers
import keras

import getData


def getModel():
    inputs = Input(shape=(getData.height, getData.width, getData.n_channel))

    merge_1 = conv_block(inputs, 3, [64, 64, 256], strides=(1, 1))
    # x = identity_block(x, 3, [64, 64, 256])
    # merge_1 = identity_block(x, 3, [64, 64, 256])

    merge_2 = conv_block(merge_1, 3, [128, 128, 512])
    # x = identity_block(x, 3, [128, 128, 512])
    # merge_2 = identity_block(x, 3, [128, 128, 512])

    merge_3 = conv_block(merge_2, 3, [256, 256, 1024])
    # x = identity_block(x, 3, [256, 256, 1024])
    # merge_3 = identity_block(x, 3, [256, 256, 1024])

    merge_4 = conv_block(merge_3, 3, [512, 512, 2048])
    # x = identity_block(x, 3, [512, 512, 2048])
    # merge_4 = identity_block(x, 3, [512, 512, 2048])
    merge_4 = Dropout(0.5)(merge_4)

    x = conv_block(merge_4, 3, [1024, 1024, 4096])
    # x = identity_block(x, 3, [1024, 1024, 4096])
    # x = identity_block(x, 3, [1024, 1024, 4096])
    x = Dropout(0.5)(x)

    x = de_conv_block(x, 3, [4096, 1024, 1024])
    # x = identity_block(x, 3, [4096, 1024, 1024])
    # x = identity_block(x, 3, [4096, 1024, 1024])
    x = de_conv_block(x, 3, [2048, 512, 512], merge_4)

    x = de_conv_block(x, 3, [1024, 256, 256], merge_3)
    # x = identity_block(x, 3, [1024, 256, 256])
    # x = identity_block(x, 3, [1024, 256, 256])

    x = de_conv_block(x, 3, [512, 128, 128], merge_2)
    # x = identity_block(x, 3, [512, 128, 128])
    # x = identity_block(x, 3, [512, 128, 128])

    x = de_conv_block_end(x, 3, [256, 64, 64], merge_1)
    # x = identity_block(x, 3, [256, 64, 64])
    # x = identity_block(x, 3, [256, 64, 64])


    x = Conv2D(getData.n_class, (1, 1))(x)

    output = Activation(activation='softmax')(x)

    model = Model(inputs=inputs, outputs=output)

    model.summary()

    adam = Adam(lr=1e-4)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def identity_block(input_tensor, kernel_size, filters):
    filters1, filters2, filters3 = filters

    bn_axis = 3

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               strides=(2, 2)):
    filters1, filters2, filters3 = filters

    bn_axis = 3

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)

    # print("x_11_shape,",x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal')(input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis)(shortcut)
    # print("x_12_shape",shortcut)

    x = layers.add([x, shortcut])
    # print("x_13_shape:",x)
    x = layers.Activation('relu')(x)
    return x


def de_conv_block(input_tensor,
                  kernel_size,
                  filters,
                  strides=None,
                  merge=None):
    filters1, filters2, filters3 = filters

    bn_axis = 3

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(input_tensor))
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation('relu')(x)

    if merge != None:
        x = concatenate([x, merge], axis=3)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    print("x_shape:", x)

    shortcut = layers.Conv2D(filters3, (1, 1),
                             kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(input_tensor))
    shortcut = layers.BatchNormalization(axis=bn_axis)(shortcut)

    print("shortcut_shape:", shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def de_conv_block_end(input_tensor,
                  kernel_size,
                  filters,
                  strides=None,
                  merge=None):
    filters1, filters2, filters3 = filters

    bn_axis = 3

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation('relu')(x)

    if merge != None:
        x = concatenate([x, merge], axis=3)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    print("x_shape:", x)

    shortcut = layers.Conv2D(filters3, (1, 1),
                             kernel_initializer='he_normal')(input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis)(shortcut)

    print("shortcut_shape:", shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

# def bottleneck():
#     x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
#     x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
#     x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
#
#     x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
#
#     x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
#
#     x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
#     x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
#     x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
