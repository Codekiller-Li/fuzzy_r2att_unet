import os
import sys

import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

import getData
import Unet
import UnetCrf
import FuzzyUnet
import UnetSoftmax

test_num = 50
model_path = "../Model/"
model_name = "temp"
x_test_path = ""
y_test_path = ""
target_class = 1
batch_size = 10
predict_file_path = "../Result/"


def demo(x_test_path=x_test_path, y_test_path=y_test_path, model_path=model_path, target_class=target_class):
    x_test_path ="../ImgData/TestT/"
    y_test_path="../ImgData/ResultT/"

    # elif sys.argv[3] == "F":
    #     x_test_path ="../ImgData/TestF/"
    #     y_test_path="../ImgData/ResultF/"

    # else:
    #     print("输入错误，重新输入")

    x_test_list = os.listdir(x_test_path)
    y_test_list = os.listdir(y_test_path)

    file_num = len(x_test_list)

    x_test_list.sort()
    y_test_list.sort()

    x_test = np.ndarray((file_num, data.height, data.width, data.n_channel), dtype=np.float)
    y_test = np.ndarray([file_num, data.height, data.width, data.n_class], dtype=np.float)

    for i, x_test_name in enumerate(x_test_list):
        img = image.load_img(x_test_path + x_test_name, target_size=(data.height, data.width))
        x_test[i] = image.img_to_array(img)

    for i, y_test_name in enumerate(y_test_list):
        img = image.load_img(y_test_path + y_test_name, target_size=(data.height, data.width))
        data = image.img_to_array(img)
        data = data[:, :, 0]
        temp = to_categorical(data.flatten(), data.n_class)
        y_test[i] = temp.reshape(data.height, data.width, data.n_class)

    if file_num > test_num:
        x_test = x_test[:test_num, :, :, :]
        y_test = y_test[:test_num, :, :, :]

    # 0~1归一化
    x_test /= 255

    # -1～1归一化
    # x_test = x_test/127.5-1

    # rgb-mean归一化
    # r_mean_value = np.mean(x_test[:, :, :, 0])
    # g_mean_value = np.mean(x_test[:, :, :, 1])
    # b_mean_value = np.mean(x_test[:, :, :, 2])
    # x_test[:, :, :, 0] = x_test[:, :, :, 0] - r_mean_value
    # x_test[:, :, :, 1] = x_test[:, :, :, 1] - g_mean_value
    # x_test[:, :, :, 2] = x_test[:, :, :, 2] - b_mean_value

    # if sys.argv[1] == "-unet":
    #     print("model_name : {}".format("unet"))
    model = Unet.getModel()

    # elif sys.argv[1] == "-crfunet":
    #     print("model_name : {}".format("crfunet"))
    #     model = UnetCrf.getModel()
    #
    # elif sys.argv[1] == "-fuzzyunet":
    #     print("model_name : {}".format("fuzzyunet"))
    #     model = FuzzyUnet.getModel()
    #
    # elif sys.argv[1] == "-softmaxunet":
    #     print("model_name : {}".format("softmaxunet"))
    #     model = UnetSoftmax.getModel()
    #
    # else:
    #     print("输入错误，重新输入")

    model_name = "unet_temp_batchsize_10_epochs_60_1209_12_08.h5"

    model_path = model_path+model_name

    model_name=str(model_name).split('.')[0]

    model.load_weights(model_path)

    y_predict = model.predict(x_test, batch_size=batch_size, verbose=0)

    y_predict = y_predict.reshape(x_test.shape[0], data.height * data.width, data.n_class)

    y_predict = np.argmax(y_predict, axis=2)

    columns = ['TP', 'FN', 'FP', 'TN', 'IoU/Jaccard', 'Precision', 'Recall', 'F1_measure']

    index = x_test_list[:x_test.shape[0]]

    index.append("Average")

    predict_data = np.ndarray(shape=(x_test.shape[0], len(columns)), dtype=float)

    y_true = np.ndarray((data.height *  data.width, 1), dtype=np.float)
    y_pred = np.ndarray((data.height *  data.width, 1), dtype=np.float)

    y_test = y_test.reshape(x_test.shape[0],data.height * data.width,data.n_class)

    for i in range(x_test.shape[0]):
        y_predict_temp = to_categorical(y_predict[i], data.n_class)
        y_true[:, 0] = y_test[i, :, target_class]
        y_pred[:, 0] = y_predict_temp[:, target_class]

        if np.sum(y_true == 0) == data.height * data.width and np.sum(
                y_pred == 0) == data.height * data.width:
            TN, FP, FN, TP = data.height * data.width, 0, 0, 0
        else:
            TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()

        predict_data[i, 0] = TP
        predict_data[i, 1] = FN
        predict_data[i, 2] = FP
        predict_data[i, 3] = TN

        if TP == 0 and FN == 0 and FP == 0:
            Iou = 0
            precision = 0
            recall = 0
            F1_measure = 0
        else:
            Iou = TP / (TP + FN + FP)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F1_measure = 2 * TP / (2 * TP + FP + FN)

        predict_data[i, 4] = Iou
        predict_data[i, 5] = precision
        predict_data[i, 6] = recall
        predict_data[i, 7] = F1_measure

    predict_average = np.ndarray(shape=(1, len(columns)))

    for i in range(predict_data.shape[1]):
        predict_average[0, i] = np.mean(predict_data[:, i])

    predict_result = np.ndarray(shape=(x_test.shape[0] + 1, len(columns)), dtype=float)

    predict_result[:x_test.shape[0], :] = predict_data
    predict_result[x_test.shape[0]:, :] = predict_average

    predict_dir = os.path.split(predict_file_path)[0]
    if not os.path.isdir(predict_dir):
        os.makedirs(predict_dir)

    df = pd.DataFrame(predict_result, columns=columns, index=index)
    df.to_csv(predict_file_path + "channel_{}_class_{}_{}_num_{}_model_{}.csv".format(data.n_channel, data.n_class, "T",x_test.shape[0], model_name))


def evaluate(x_test_path=x_test_path, y_test_path=y_test_path, model_path=model_path, target_class=target_class):
    if sys.argv[3] == "T":
        x_test_path ="../ImgData/TestT/"
        y_test_path="../ImgData/ResultT/"

    elif sys.argv[3] == "F":
        x_test_path ="../ImgData/TestF/"
        y_test_path="../ImgData/ResultF/"

    else:
        print("输入错误，重新输入")

    x_test_list = os.listdir(x_test_path)
    y_test_list = os.listdir(y_test_path)

    file_num = len(x_test_list)

    x_test_list.sort()
    y_test_list.sort()

    x_test = np.ndarray((file_num, data.height, data.width, data.n_channel), dtype=np.float)
    y_test = np.ndarray([file_num, data.height, data.width, data.n_class], dtype=np.float)

    for i, x_test_name in enumerate(x_test_list):
        img = image.load_img(x_test_path + x_test_name, target_size=(data.height, data.width))
        x_test[i] = image.img_to_array(img)

    for i, y_test_name in enumerate(y_test_list):
        img = image.load_img(y_test_path + y_test_name, target_size=(data.height, data.width))
        data = image.img_to_array(img)
        data = data[:, :, 0]
        temp = to_categorical(data.flatten(), data.n_class)
        y_test[i] = temp.reshape(data.height, data.width, data.n_class)

    if file_num > test_num:
        x_test = x_test[:test_num, :, :, :]
        y_test = y_test[:test_num, :, :, :]

    # 0~1归一化
    x_test /= 255

    # -1～1归一化
    # x_test = x_test/127.5-1

    # rgb-mean归一化
    # r_mean_value = np.mean(x_test[:, :, :, 0])
    # g_mean_value = np.mean(x_test[:, :, :, 1])
    # b_mean_value = np.mean(x_test[:, :, :, 2])
    # x_test[:, :, :, 0] = x_test[:, :, :, 0] - r_mean_value
    # x_test[:, :, :, 1] = x_test[:, :, :, 1] - g_mean_value
    # x_test[:, :, :, 2] = x_test[:, :, :, 2] - b_mean_value

    if sys.argv[1] == "-unet":
        print("model_name : {}".format("unet"))
        model = Unet.getModel()

    elif sys.argv[1] == "-crfunet":
        print("model_name : {}".format("crfunet"))
        model = UnetCrf.getModel()

    elif sys.argv[1] == "-fuzzyunet":
        print("model_name : {}".format("fuzzyunet"))
        model = FuzzyUnet.getModel()

    elif sys.argv[1] == "-softmaxunet":
        print("model_name : {}".format("softmaxunet"))
        model = UnetSoftmax.getModel()

    else:
        print("输入错误，重新输入")

    model_path = model_path+sys.argv[2]

    model_name=str(sys.argv[2]).split('.')[0]

    model.load_weights(model_path)

    y_predict = model.predict(x_test, batch_size=batch_size, verbose=0)

    y_predict = y_predict.reshape(x_test.shape[0], data.height * data.width, data.n_class)

    y_predict = np.argmax(y_predict, axis=2)

    columns = ['TP', 'FN', 'FP', 'TN', 'IoU/Jaccard', 'Precision', 'Recall', 'F1_measure']

    index = x_test_list[:x_test.shape[0]]

    index.append("Average")

    predict_data = np.ndarray(shape=(x_test.shape[0], len(columns)), dtype=float)

    y_true = np.ndarray((data.height *  data.width, 1), dtype=np.float)
    y_pred = np.ndarray((data.height *  data.width, 1), dtype=np.float)

    y_test = y_test.reshape(x_test.shape[0],data.height * data.width,data.n_class)

    for i in range(x_test.shape[0]):
        y_predict_temp = to_categorical(y_predict[i], data.n_class)
        y_true[:, 0] = y_test[i, :, target_class]
        y_pred[:, 0] = y_predict_temp[:, target_class]

        if np.sum(y_true == 0) == data.height * data.width and np.sum(
                y_pred == 0) == data.height * data.width:
            TN, FP, FN, TP = data.height * data.width, 0, 0, 0
        else:
            TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()

        predict_data[i, 0] = TP
        predict_data[i, 1] = FN
        predict_data[i, 2] = FP
        predict_data[i, 3] = TN

        if TP == 0 and FN == 0 and FP == 0:
            Iou = 0
            precision = 0
            recall = 0
            F1_measure = 0
        else:
            Iou = TP / (TP + FN + FP)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F1_measure = 2 * TP / (2 * TP + FP + FN)

        predict_data[i, 4] = Iou
        predict_data[i, 5] = precision
        predict_data[i, 6] = recall
        predict_data[i, 7] = F1_measure

    predict_average = np.ndarray(shape=(1, len(columns)))

    for i in range(predict_data.shape[1]):
        predict_average[0, i] = np.mean(predict_data[:, i])

    predict_result = np.ndarray(shape=(x_test.shape[0] + 1, len(columns)), dtype=float)

    predict_result[:x_test.shape[0], :] = predict_data
    predict_result[x_test.shape[0]:, :] = predict_average

    predict_dir = os.path.split(predict_file_path)[0]
    if not os.path.isdir(predict_dir):
        os.makedirs(predict_dir)

    df = pd.DataFrame(predict_result, columns=columns, index=index)
    df.to_csv(predict_file_path + "channel_{}_class_{}_{}_num_{}_model_{}.csv".format(data.n_channel, data.n_class, str(sys.argv[3]),x_test.shape[0], model_name))

if __name__ == '__main__':
    # evaluate()
    demo()
