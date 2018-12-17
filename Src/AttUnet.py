from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation
from keras.layers import BatchNormalization, add, multiply,PReLU
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def conv_block(nbFilters,filtersize,x):
    x = Conv2D(nbFilters, (filtersize, filtersize), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(nbFilters, (filtersize, filtersize), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def up_conv_block(nbFilters, x):
    x = Conv2DTranspose(nbFilters, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def Attention_block(nbFilters, g, x):
    g1 = Conv2D(nbFilters, (1, 1), padding='same')(g)
    g1 = BatchNormalization()(g1)

    x1 = Conv2D(nbFilters, (1, 1), padding='same')(x)
    x1 = BatchNormalization()(x1)

    psi = Activation('relu')(add([g1, x1]))
    psi = Conv2D(1, (1, 1), padding='same')(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)

    x = multiply([psi, x])
    return x

def Recurrent_block(nbFilters, t, x):

    for i in range(t):
        if i == 0:
            x1 = Conv2D(nbFilters, (3, 3), padding='same')(x)
            x1 = BatchNormalization()(x1)
            x1 = PReLU()(x1)
        x1 = Conv2D(nbFilters, (3, 3), padding='same')(add([x, x1]))
        x1 = BatchNormalization()(x1)
        x1 = PReLU()(x1)

    return x1

def RRCNN_block(nbFilters, t, x):
    x1 = Conv2D(nbFilters, (1, 1), padding='same')(x)
    x2 = Recurrent_block(nbFilters, t, x1)
    x3 = Recurrent_block(nbFilters, t, x2)
    return x3

class AttU_Net(object):
    def __init__(self, input_shape, model_file_path="./Model/"):
        self.input_shape = input_shape
        self.model_file_path = model_file_path

    def create(self,num_classes, lr_init, lr_decay):
        img_input = Input(self.input_shape)

        x1 = conv_block(64, 3, img_input)

        x2 = MaxPooling2D()(x1)
        x2 = conv_block(128, 3, x2)

        x3 = MaxPooling2D()(x2)
        x3 = conv_block(256, 3, x3)

        x4 = MaxPooling2D()(x3)
        x4 = conv_block(512, 3, x4)

        x5 = MaxPooling2D()(x4)
        x5 = conv_block(1024, 3, x5)

        d5 = up_conv_block(512, x5)
        x4 = Attention_block(256, d5, x4)
        d5 = concatenate([x4, d5])
        d5 = conv_block(512, 3, d5)

        d4 = up_conv_block(256, d5)
        x3 = Attention_block(128, d4, x3)
        d4 = concatenate([x3, d4])
        d4 = conv_block(256, 3, d4)

        d3 = up_conv_block(128, d4)
        x2 = Attention_block(64, d3, x2)
        d3 = concatenate([x2, d3])
        d3 = conv_block(128, 3, d3)

        d2 = up_conv_block(64, d3)
        x1 = Attention_block(32, d2, x1)
        d2 = concatenate([x1, d2])
        d2 = conv_block(64, 3, d2)

        d1 = Conv2D(num_classes, (1, 1), activation='softmax',padding='same')(d2)

        model = Model(img_input, d1)

        model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

        return model

class R2U_Net(object):
    def __init__(self, input_shape, model_file_path="./Model/"):
        self.input_shape = input_shape
        self.model_file_path = model_file_path

    def create(self,num_classes, lr_init, lr_decay, t):
        img_input = Input(self.input_shape)
        x1 = RRCNN_block(64, t, img_input)

        x2 = MaxPooling2D()(x1)
        x2 = RRCNN_block(128, t, x2)

        x3 = MaxPooling2D()(x2)
        x3 = RRCNN_block(256, t, x3)

        x4 = MaxPooling2D()(x3)
        x4 = RRCNN_block(512, t, x4)

        x5 = MaxPooling2D()(x4)
        x5 = RRCNN_block(1024, t, x5)

        d5 = up_conv_block(512, x5)
        d5 = concatenate([x4, d5])
        d5 = RRCNN_block(512, t, d5)

        d4 = up_conv_block(256, d5)
        d4 = concatenate([x3, d4])
        d4 = RRCNN_block(256, t, d4)

        d3 = up_conv_block(128, d4)
        d3 = concatenate([x2, d3])
        d3 = RRCNN_block(128, t, d3)

        d2 = up_conv_block(64, d3)
        d2 = concatenate([x1, d2])
        d2 = RRCNN_block(64, t, d2)

        d1 = Conv2D(num_classes, (1, 1), activation='softmax',padding='same')(d2)

        model = Model(img_input, d1)

        model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        return model

class R2AttU_Net(object):
    def __init__(self, input_shape, model_file_path="./Model/"):
        self.input_shape = input_shape
        self.model_file_path = model_file_path

    def create(self,num_classes, lr_init, lr_decay, t):
        img_input = Input(self.input_shape)
        x1 = RRCNN_block(64, t, img_input)

        x2 = MaxPooling2D()(x1)
        x2 = RRCNN_block(128, t, x2)

        x3 = MaxPooling2D()(x2)
        x3 = RRCNN_block(256, t, x3)

        x4 = MaxPooling2D()(x3)
        x4 = RRCNN_block(512, t, x4)

        x5 = MaxPooling2D()(x4)
        x5 = RRCNN_block(1024, t, x5)

        d5 = up_conv_block(512, x5)
        x4 = Attention_block(256, d5, x4)
        d5 = concatenate([x4, d5])
        d5 = RRCNN_block(512, t, d5)

        d4 = up_conv_block(256, d5)
        x3 = Attention_block(128, d4, x3)
        d4 = concatenate([x3, d4])
        d4 = RRCNN_block(256, t, d4)

        d3 = up_conv_block(128, d4)
        x2 = Attention_block(64, d3, x2)
        d3 = concatenate([x2, d3])
        d3 = RRCNN_block(128, t, d3)

        d2 = up_conv_block(64, d3)
        x1 = Attention_block(32, d2, x1)
        d2 = concatenate([x1, d2])
        d2 = RRCNN_block(64, t, d2)

        d1 = Conv2D(num_classes, (1, 1), activation='softmax',padding='same')(d2)

        model = Model(img_input, d1)

        model.summary()

        model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

        return model

if __name__ == '__main__':
    m1 = AttU_Net((256, 256, 3)).create(5, 1e-4, 5e-4,2)
    m1.summary()
    m2 = R2U_Net((256, 256, 3)).create(5, 1e-4, 5e-4,2)
    m2.summary()
    m3 = R2AttU_Net((256, 256, 3)).create(5, 1e-4, 5e-4,2)
    m3.summary()
