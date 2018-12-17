from keras.layers import *
from keras.models import Model
from keras.optimizers import *
import getData
import util_softmax


def getModel():
    inputs = Input(shape=(getData.height, getData.width, getData.n_channel))

    # multi方法：
    fuzzy = FuzzyMultiLayer((getData.height, getData.width, getData.n_class))(inputs)
    fuzzy = BatchNormalization(epsilon=1e-6, momentum=0.99)(fuzzy)

    # merge方法：
    # fuzzy = FuzzyMergeLayer((getData.height, getData.width, getData.n_class))(inputs)
    # fuzzy_softmax = tf.slice(fuzzy, [0, 0, 0, getData.n_channel], [-1, -1, -1, getData.n_class])
    # fuzzy = BatchNormalization(epsilon=1e-6, momentum=0.99)(fuzzy)


    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(fuzzy)
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

    # model.compile(optimizer=adam, loss=util_softmax.SoftmaxLoss(fuzzy).softmaxLoss(), metrics=['accuracy'])

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


tfd = tf.contrib.distributions


class FuzzyMultiLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(FuzzyMultiLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=(getData.n_class, 1, getData.n_channel, getData.n_channel),
                                     initializer='uniform', trainable=True)
        self.mean = self.add_weight(name='mean', shape=(getData.n_class, 1, getData.n_channel), initializer='uniform',
                                    trainable=True)
        self.scale = tf.tile(self.scale, multiples=[1, getData.height * getData.width, 1, 1])
        self.mean = tf.tile(self.mean, multiples=[1, getData.height * getData.width, 1])
        super(FuzzyMultiLayer, self).build(input_shape)

    def call(self, x):
        x = tf.reshape(x, [-1, getData.height * getData.width, getData.n_channel])
        gauss = []
        output = []
        for i in range(getData.n_class):
            mvn = tfd.MultivariateNormalTriL(
                loc=self.mean[i],
                scale_tril=self.scale[i])
            gauss_tensor = mvn.prob(x)
            gauss.append(gauss_tensor)

        temp = tf.convert_to_tensor(gauss)
        temp = tf.transpose(temp, perm=[1, 2, 0])

        gauss = tf.reshape(temp, [-1, getData.height, getData.width, getData.n_class])
        gauss = tf.nn.l2_normalize(gauss, dim=3)

        for i in range(getData.n_class):
            gauss_tensor = tf.reshape(tf.slice(gauss, [0, 0, 0, i], [-1, -1, -1, 1]), [-1, getData.height * getData.width])
            for j in range(getData.n_channel):
                temp = tf.multiply(gauss_tensor, tf.reshape(tf.slice(x, [0, 0, j], [-1, -1, 1]),[-1,getData.height * getData.width]))
                output.append(temp)

        temp = tf.convert_to_tensor(output)
        temp = tf.transpose(temp, perm=[1, 2, 0])
        output = tf.reshape(temp, [-1, getData.height, getData.width, getData.n_class * getData.n_channel])
        return output

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.output_dim[0], self.output_dim[1], getData.n_class * getData.n_channel)
        return output_shape




class FuzzyMergeLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(FuzzyMergeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=(getData.n_class, 1, getData.n_channel, getData.n_channel),
                                     initializer='uniform', trainable=True)
        self.mean = self.add_weight(name='mean', shape=(getData.n_class, 1, getData.n_channel), initializer='uniform',
                                    trainable=True)
        self.scale = tf.tile(self.scale, multiples=[1, getData.height * getData.width, 1, 1])
        self.mean = tf.tile(self.mean, multiples=[1, getData.height * getData.width, 1])

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

        temp = tf.convert_to_tensor(output)
        temp = tf.transpose(temp, perm=[1, 2, 0])

        output = tf.reshape(temp, [-1, getData.height, getData.width, getData.n_class])
        output = tf.nn.l2_normalize(output, dim=3)
        return concatenate([tf.reshape(x,[-1,getData.height,getData.width,getData.n_channel]), output], axis=3)

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.output_dim[0], self.output_dim[1], getData.n_class+getData.n_channel)
        return output_shape



if __name__ == '__main__':
    print("1")
    # batch_size = 2
    # class = 5
    # channel = 3
    # width = 2
    # height = 2

    # 验证高斯公式
    # x = tf.Variable(tf.zeros([2, 4, 3]), name='x')
    # scale = tf.Variable(tf.ones([4, 3, 3]), name='scale')
    # loc = tf.Variable(tf.zeros([4, 3]), name='loc')
    #
    # mvn = tfd.MultivariateNormalTriL(
    #     loc=loc,
    #     scale_tril=scale)
    # gauss = mvn.prob(x)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print(sess.run(x))
    #     print(sess.run(gauss))

    # 验证tf.convert_to_tensor
    # x = tf.Variable(np.array([[[1, 0, 0],
    #                            [0, 0, 0],
    #                            [0, 0, 0],
    #                            [0, 0, 0]],
    #
    #                           [[0, 0, 0],
    #                            [0, 0, 0],
    #                            [0, 0, 0],
    #                            [0, 0, 0]]]), name='x', dtype='float32')
    #
    # scale = tf.Variable(tf.ones([5, 4, 3, 3]), name='scale')
    # loc = tf.Variable(tf.zeros([5, 4, 3]), name='loc')
    #
    # output = []
    # for i in range(5):
    #     mvn = tfd.MultivariateNormalTriL(
    #         loc=loc[i],
    #         scale_tril=scale[i])
    #     gauss = mvn.prob(x)
    #     output.append(gauss)
    # x3 = tf.convert_to_tensor(output)
    # print("x3",x3)
    # x1 = tf.transpose(x3, perm=[1, 2, 0])
    #
    # x2 = tf.reshape(x1,[-1, 4, 5])
    # print("x2",x2)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print(sess.run(x))
    #
    #     print(sess.run(x3))
    #     print(sess.run(x1))
    #     print(sess.run(x2))
