import tensorflow as tf
from keras.backend import epsilon
from keras.layers import *
from keras.models import Model
from keras.optimizers import *
import getData


class MySoftmax(Layer):
    def __init__(self, **kwargs):
        super(MySoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MySoftmax, self).build(input_shape)

    def call(self, x):
        print("type", type(x))
        e = K.exp(x - K.max(x, axis=3, keepdims=True))
        print("e_shape:", e)
        s = K.sum(e, axis=3, keepdims=True)
        print("s_shape:", s)
        print("e/s_shape:", e / s)
        # x=tf.exp(x,name="exp")
        # print("x_shape:",tf.reduce_sum(x,axis=3))
        # print("x_shape",x)
        return e / s

    def compute_output_shape(self, input_shape):
        return input_shape


class SoftmaxLoss(Layer):
    def __init__(self, fuzzy_tensor):
        self.fuzzy_tensor = fuzzy_tensor

    def softmaxLoss(self):
        def getLoss(y_true, y_pred):
            y_pred /= tf.reduce_sum(y_pred, -1, True)
            _epsilon = tf.convert_to_tensor(epsilon(), dtype=float)
            y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)

            param_1 = tf.ones_like(self.fuzzy_tensor)
            param_2 = tf.ones_like(self.fuzzy_tensor)*1.2
            temp = tf.where(self.fuzzy_tensor > 0.6, param_1, param_2)
            self.fuzzy_tensor = tf.where(self.fuzzy_tensor < 0.4, param_1, temp)


            # temp = tf.tile(self.fuzzy_tensor, multiples=[1, 1, 1, 2])
            #
            # param_1 = tf.slice(temp, [0, 0, 0, 0], [-1, -1, -1, data.n_class])
            # param_1 = tf.clip_by_value(param_1, 1., 1.)
            #
            # param_2 = tf.slice(temp, [0, 0, 0, data.n_class], [-1, -1, -1, data.n_class])
            # param_2 = tf.clip_by_value(param_2, 1.2, 1.2)
            #
            # self.fuzzy_tensor = tf.where(self.fuzzy_tensor > 0.6, param_1, self.fuzzy_tensor)
            # self.fuzzy_tensor = tf.where(self.fuzzy_tensor < 0.4, param_1, self.fuzzy_tensor)
            # self.fuzzy_tensor = tf.where(tf.equal(self.fuzzy_tensor, 1), param_1, param_2)

            loss = - tf.reduce_sum(self.fuzzy_tensor * y_true * tf.log(y_pred), -1)
            return loss

        return getLoss


# def softmaxLoss(fuzzy_tensor):
#     def getLoss(y_true, y_pred):
#         print("fuzzy_tensor_shape:", fuzzy_tensor)
#         temp = fuzzy_tensor
#         y_pred /= tf.reduce_sum(y_pred, -1, True)
#         _epsilon = tf.convert_to_tensor(epsilon(), dtype=float)
#         y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
#         loss = - tf.reduce_sum(y_true * tf.log(y_pred), -1)
#
#         # fuzzy_tensor=tf.Session().run(fuzzy_tensor)
#         # fuzzy_tensor[((fuzzy_tensor>0.4) & (fuzzy_tensor<0.6))]=1.2
#         # fuzzy_tensor[fuzzy_tensor != 1.2]=1
#         # fuzzy_tensor=tf.convert_to_tensor(fuzzy_tensor)
#         # loss = fuzzy_tensor*loss
#         return loss
#
#     return getLoss
