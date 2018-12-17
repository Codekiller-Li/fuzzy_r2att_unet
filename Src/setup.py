# -*- coding: UTF-8 -*-
import datetime
import sys

from keras.callbacks import ModelCheckpoint
from keras.layers import *

import getData
import UnetSoftmax
import Unet
import FuzzyUnet
import ResUnet
import AttUnet

model_file_path = "../Model/"


def main():
    batch_size = int(sys.argv[2])
    epochs = int(sys.argv[3])

    x_train = np.load(getData.x_train_path).astype('float32')
    y_train = np.load(getData.y_train_path).astype('float32')

    # 0~1归一化
    x_train /= 255

    # -1～1归一化
    # x_train = x_train/127.5-1

    # rgb-mean归一化
    # r_mean_value = np.mean(x_train[:, :, :, 0])
    # g_mean_value = np.mean(x_train[:, :, :, 1])
    # b_mean_value = np.mean(x_train[:, :, :, 2])
    # x_train[:, :, :, 0] = x_train[:, :, :, 0] - r_mean_value
    # x_train[:, :, :, 1] = x_train[:, :, :, 1] - g_mean_value
    # x_train[:, :, :, 2] = x_train[:, :, :, 2] - b_mean_value

    nowTime = datetime.datetime.now().strftime('%m%d_%H_%M')

    if sys.argv[1] == "-unet":
        print("model_name : {}".format("unet"))
        model = Unet.getModel()
        model_temp_name = "unet_temp_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
        model_name = "unet_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
        run(model, batch_size, epochs, model_temp_name, model_name, x_train, y_train)

    elif sys.argv[1] == "-fuzzyunet":
        print("model_name : {}".format("fuzzyunet"))
        model = FuzzyUnet.getModel()
        model_temp_name = "fuzzy_unet_temp_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
        model_name = "fuzzy_unet_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
        run(model, batch_size, epochs, model_temp_name, model_name, x_train, y_train)

    elif sys.argv[1] == "-softmaxunet":
        print("model_name : {}".format("softmaxunet"))
        model = UnetSoftmax.getModel()
        model_temp_name = "softmax_unet_temp_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
        model_name = "softmax_unet_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
        run(model, batch_size, epochs, model_temp_name, model_name, x_train, y_train)

    elif sys.argv[1] == "-resunet":
        print("model_name : {}".format("resunet"))
        model = ResUnet.getModel()
        model_temp_name = "res_unet_temp_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
        model_name = "res_unet_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
        run(model, batch_size, epochs, model_temp_name, model_name, x_train, y_train)

    elif sys.argv[1] == "-attunet":
        print("model_name : {}".format("attunet"))
        model = AttUnet.AttU_Net((getData.height, getData.width, getData.n_channel)).create(getData.n_class, 1e-4, 5e-4, 2)
        model_temp_name = "attunet_temp_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
        model_name = "attunet_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
        run(model, batch_size, epochs, model_temp_name, model_name, x_train, y_train)

    elif sys.argv[1] == "-r2unet":
        print("model_name : {}".format("r2unet"))
        model = AttUnet.R2U_Net((getData.height, getData.width, getData.n_channel)).create(getData.n_class, 1e-4, 5e-4, 2)
        model_temp_name = "r2unet_temp_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
        model_name = "r2unet_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
        run(model, batch_size, epochs, model_temp_name, model_name, x_train, y_train)

    elif sys.argv[1] == "-r2attunet":
        print("model_name : {}".format("r2attunet"))
        model = AttUnet.R2AttU_Net((getData.height, getData.width, getData.n_channel)).create(getData.n_class, 1e-4, 5e-4, 2)
        model_temp_name = "r2attunet_temp_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
        model_name = "r2attunet_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
        run(model, batch_size, epochs, model_temp_name, model_name, x_train, y_train)

    else:
        print("输入错误，重新输入")


def run(model=None,
        batch_size=None,
        epochs=None,
        model_temp_name=None,
        model_name=None,
        x_train=None,
        y_train=None):
    model_checkpoint = ModelCheckpoint(model_file_path + model_temp_name, monitor='val_loss', verbose=0,
                                       save_best_only=True,
                                       save_weights_only=False, mode='auto', period=1)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1,
              shuffle=True,
              callbacks=[model_checkpoint])

    model.fit_generator()

    # score = model.evaluate(x_test, y_test, verbose=0)
    #
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])

    model.save(model_file_path + model_name)


def demo():
    batch_size = 1
    epochs = 10

    x_train = np.load(getData.x_train_path).astype('float32')
    y_train = np.load(getData.y_train_path).astype('float32')

    print("x_train:",x_train.shape)
    print("y_train:",y_train.shape)

    x_train = x_train[:10]
    y_train = y_train[:10]

    # 0~1归一化
    x_train /= 255

    # -1～1归一化
    # x_train = x_train/127.5-1

    # rgb-mean归一化
    # r_mean_value = np.mean(x_train[:, :, :, 0])
    # g_mean_value = np.mean(x_train[:, :, :, 1])
    # b_mean_value = np.mean(x_train[:, :, :, 2])
    # x_train[:, :, :, 0] = x_train[:, :, :, 0] - r_mean_value
    # x_train[:, :, :, 1] = x_train[:, :, :, 1] - g_mean_value
    # x_train[:, :, :, 2] = x_train[:, :, :, 2] - b_mean_value

    nowTime = datetime.datetime.now().strftime('%m%d_%H_%M')

    # softmax_unet:
    # model = UnetSoftmax.getModel()
    # model_temp_name = "softmax_unet_temp_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
    # model_name = "softmax_unet_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
    # run(model, batch_size, epochs, model_temp_name, model_name, x_train, y_train)

    # fuzzy_unet:
    model = FuzzyUnet.getModel()
    model_temp_name = "fuzzy_unet_temp_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
    model_name = "fuzzy_unet_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
    run(model, batch_size, epochs, model_temp_name, model_name, x_train, y_train)

    # unet:
    # model = Unet.getModel()
    # model_temp_name = "unet_temp_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
    # model_name = "unet_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
    # run(model, batch_size, epochs, model_temp_name, model_name, x_train, y_train)

    # res_unet:
    # model = ResUnet.getModel()
    # model_temp_name = "res_unet_temp_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
    # model_name = "res_unet_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
    # run(model, batch_size, epochs, model_temp_name, model_name, x_train, y_train)

    # rcnn:
    # model = Keras_RCNN.getModel()
    # model_temp_name = "rcnn_temp_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
    # model_name = "rcnn_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
    # run(model, batch_size, epochs, model_temp_name, model_name, x_train, y_train)

    #AttUnet:
    # model = AttUnet.AttU_Net((getData.height, getData.width, getData.n_channel)).create(getData.n_class, 1e-4, 5e-4, 2)
    # model_temp_name = "attunet_temp_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
    # model_name = "attunet_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
    # run(model, batch_size, epochs, model_temp_name, model_name, x_train, y_train)

    #R2Unet:
    # model = AttUnet.R2U_Net((getData.height, getData.width, getData.n_channel)).create(getData.n_class, 1e-4, 5e-4, 2)
    # model_temp_name = "r2unet_temp_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
    # model_name = "r2unet_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
    # run(model, batch_size, epochs, model_temp_name, model_name, x_train, y_train)

    #R2AttUnet:
    # model = AttUnet.R2AttU_Net((getData.height, getData.width, getData.n_channel)).create(getData.n_class, 1e-4, 5e-4, 2)
    # model_temp_name = "r2attunet_temp_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
    # model_name = "r2attunet_batchsize_{}_epochs_{}_{}.h5".format(batch_size, epochs, nowTime)
    # run(model, batch_size, epochs, model_temp_name, model_name, x_train, y_train)


if __name__ == '__main__':
    demo()
    # main()
