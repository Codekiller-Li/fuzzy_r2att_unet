from keras.layers import *
from keras.models import Model
from keras.optimizers import *
import getData
import util_softmax


def getModel():
    inputs = Input(shape=(getData.height, getData.width, getData.n_channel))

    # multi方法：
    # fuzzy = FuzzyMultiLayer((data.height, data.width, data.n_class))(inputs)
    # fuzzy = BatchNormalization(epsilon=1e-6, momentum=0.99)(fuzzy)

    # merge_1方法：
    fuzzy = FuzzyMergeLayer((getData.height, getData.width, getData.n_class))(inputs)
    fuzzy_merge = concatenate([inputs, fuzzy], axis=3)
    #fuzzy_loss = tf.slice(fuzzy, [0, 0, 0, data.n_channel], [-1, -1, -1, data.n_class])
    fuzzy_merge = BatchNormalization(epsilon=1e-6, momentum=0.99)(fuzzy_merge)

    # merge_2方法:
    # fuzzy = GaussLayer((data.height, data.width, data.n_class))(inputs)
    # fuzzy = Activation(activation='softmax')(fuzzy)
    # fuzzy = concatenate([fuzzy, inputs], axis=3)

    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(fuzzy_merge)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(getData.n_class, (1, 1))(conv9)

    output = Activation(activation='softmax')(conv10)

    model = Model(inputs=inputs, outputs=output)

    model.summary()

    adam = Adam(lr=1e-4)

    model.compile(optimizer=adam, loss=util_softmax.SoftmaxLoss(fuzzy).softmaxLoss(), metrics=['accuracy'])

    # model.compile(optimizer=adam, loss=util_softmax.SoftmaxLoss(fuzzy_loss).softmaxLoss(), metrics=['accuracy'])

    return model


tfd = tf.contrib.distributions


class FuzzyMultiLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(FuzzyMultiLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(name='v1', shape=(getData.n_class, 1, getData.n_channel, getData.n_channel),
                                     initializer='uniform', trainable=True)
        self.mean = self.add_weight(name='v2', shape=(getData.n_class, 1, getData.n_channel), initializer='uniform',
                                    trainable=True)
        self.scale=tf.tile(self.scale, multiples=[1, getData.height * getData.width, 1, 1])
        self.mean=tf.tile(self.mean, multiples=[1, getData.height * getData.width, 1])
        print("mean_shape1:",self.mean)
        print("scale_shape1:",self.scale)
        super(FuzzyMultiLayer, self).build(input_shape)

    def call(self, x):
        print("type_x", type(x))
        x = tf.reshape(x, [-1, getData.height * getData.width, getData.n_channel])
        # scale_temp = tf.tile(self.scale, multiples=[1, data.height * data.width, 1, 1])
        # mean_temp = tf.tile(self.mean, multiples=[1, data.height * data.width, 1])
        print("mean_shape:",self.mean.shape)
        print("scale_shape:",self.scale.shape)

        gauss = []
        output = []
        for i in range(getData.n_class):
            mvn = tfd.MultivariateNormalTriL(
                loc=self.mean[i],
                scale_tril=self.scale[i])
            gauss_tensor = mvn.prob(x)
            gauss.append(gauss_tensor)

        gauss = tf.convert_to_tensor(gauss)



        for i in range(getData.n_class):
            gauss_tensor = tf.reshape(tf.slice(gauss, [i, 0, 0], [1, -1, -1]), [-1, getData.height * getData.width])
            for j in range(getData.n_channel):
                temp = tf.multiply(gauss_tensor, tf.reshape(tf.slice(x, [0, 0, j], [-1, -1, 1]),
                                                            [-1, getData.height * getData.width]))
                output.append(temp)
        output = tf.reshape(tf.convert_to_tensor(output),
                            [-1, getData.height, getData.width, getData.n_class * getData.n_channel])
        output = tf.nn.l2_normalize(output, dim=-1)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.output_dim[0], self.output_dim[1], getData.n_class * getData.n_channel)
        return output_shape


class FuzzyMergeLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(FuzzyMergeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(name='v1', shape=(getData.n_class, 1, getData.n_channel, getData.n_channel),
                                     initializer='uniform', trainable=True)
        self.mean = self.add_weight(name='v2', shape=(getData.n_class, 1, getData.n_channel), initializer='uniform',
                                    trainable=True)
        self.scale=tf.tile(self.scale, multiples=[1, getData.height * getData.width, 1, 1])
        self.mean=tf.tile(self.mean, multiples=[1, getData.height * getData.width, 1])

        super(FuzzyMergeLayer, self).build(input_shape)

    def call(self, x):
        x = tf.reshape(x, [-1, getData.height * getData.width, getData.n_channel])
        output = []
        for i in range(getData.n_class):
            mvn = tfd.MultivariateNormalTriL(
                loc=self.mean[i],
                scale_tril=self.scale[i])
            gauss = mvn.prob(x)
            output.append(gauss)

        output = tf.reshape(tf.convert_to_tensor(output), [-1, getData.height, getData.width, getData.n_class])
        output = tf.nn.l2_normalize(output, dim=3)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.output_dim[0], self.output_dim[1], getData.n_class)
        return output_shape


class GaussLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(GaussLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(name='v1', shape=(getData.n_class, 1, getData.n_channel, getData.n_channel),
                                     initializer='uniform', trainable=True)
        self.mean = self.add_weight(name='v2', shape=(getData.n_class, 1, getData.n_channel), initializer='uniform',
                                    trainable=True)
        super(GaussLayer, self).build(input_shape)

    def call(self, x):
        x = tf.reshape(x, [-1, getData.height * getData.width, getData.n_channel])
        scale_temp = tf.tile(self.scale, multiples=[1, getData.height * getData.width, 1, 1])
        mean_temp = tf.tile(self.mean, multiples=[1, getData.height * getData.width, 1])
        output = []
        for i in range(getData.n_class):
            mvn = tfd.MultivariateNormalTriL(
                loc=mean_temp[i],
                scale_tril=scale_temp[i])
            gauss = mvn.prob(x)
            output.append(gauss)

        return tf.reshape(tf.convert_to_tensor(output), [-1, getData.height, getData.width, getData.n_channel])

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.output_dim[0], self.output_dim[1], getData.n_class)
        return output_shape
