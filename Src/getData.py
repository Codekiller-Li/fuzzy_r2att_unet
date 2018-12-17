# -*- coding: UTF-8 -*-
import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
import cv2

height = 256
width = 256
n_channel = 3
n_class = 5
train_num = 1
test_num = 1
test_t_rate = 0.1
test_f_rate = 0.1

img_type = "png"

data_path = "../Data{}/".format(n_class)

x_train_t_path = data_path+"XTrainT/"
x_train_f_path = data_path+"XTrainF/"
y_train_t_path = data_path+"YTrainT/"
y_train_f_path = data_path+"YTrainF/"

x_train_t_csv = data_path+"train_t_name.csv"
x_train_f_csv = data_path+"train_f_name.csv"
x_test_t_csv = data_path+"test_t_name.csv"
x_test_f_csv = data_path+"test_f_name.csv"

x_aug_t_path = data_path+"XAugT/"
x_aug_f_path = data_path+"XAugF/"
y_aug_t_path = data_path+"YAugT/"
y_aug_f_path = data_path+"YAugF/"

x_test_t_path = data_path+"XTestT/"
x_test_f_path = data_path+"XTestF/"
y_test_t_path = data_path+"YTestT/"
y_test_f_path = data_path+"YTestF/"

npy_path = "../NpyData/"

x_train_path = npy_path+"train.npy"
y_train_path = npy_path+"label.npy"

data_gen = image.ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.03,
    height_shift_range=0.03,
    shear_range=0.03,
    zoom_range=0.03,
    fill_mode='constant',
    horizontal_flip=True,
    vertical_flip=True,
)


def augGenerator(height=height,
                 width=width):
    global randomList_T, randomList_F, random_list_t, random_list_f
    createFile(x_aug_t_path)
    createFile(x_aug_f_path)
    createFile(y_aug_t_path)
    createFile(y_aug_f_path)
    createFile(x_test_t_path)
    createFile(x_test_f_path)
    createFile(y_test_t_path)
    createFile(y_test_f_path)

    x_train_t_list = os.listdir(x_train_t_path)
    x_train_f_list = os.listdir(x_train_f_path)
    y_train_t_list = os.listdir(y_train_t_path)
    y_train_f_list = os.listdir(y_train_f_path)

    x_train_t_list.sort()
    x_train_f_list.sort()
    y_train_t_list.sort()
    y_train_f_list.sort()

    if x_train_t_list[0] == '.DS_Store':
        x_train_t_list.remove('.DS_Store')
    if y_train_t_list[0] == '.DS_Store':
        y_train_t_list.remove('.DS_Store')
    if x_train_f_list[0] == '.DS_Store':
        x_train_f_list.remove('.DS_Store')
    if y_train_f_list[0] == '.DS_Store':
        y_train_f_list.remove('.DS_Store')

    test_t_num = int(test_t_rate * len(x_train_t_list))
    test_f_num = int(test_f_rate * len(x_train_f_list))

    if test_t_rate > 1 or test_f_rate > 1:
        print("test num too much over 100% !")
        return
    else:
        random_list_t = random.sample(set(np.arange(len(x_train_t_list))), test_t_num)
        random_list_t = np.sort(random_list_t)

        random_list_f = random.sample(set(np.arange(len(x_train_f_list))), test_f_num)
        random_list_f = np.sort(random_list_f)

    x_train_t = np.ndarray((len(x_train_t_list) - test_t_num, height, width, n_channel),
                           dtype=np.uint8)
    x_train_f = np.ndarray((len(x_train_f_list) - test_f_num, height, width, n_channel),
                           dtype=np.uint8)
    y_train_t = np.ndarray((len(y_train_t_list) - test_t_num, height, width, 1),
                           dtype=np.uint8)
    y_train_f = np.ndarray((len(y_train_f_list) - test_f_num, height, width, 1),
                           dtype=np.uint8)
    x_test_t = np.ndarray((test_t_num, height, width, n_channel),
                          dtype=np.uint8)
    x_test_f = np.ndarray((test_f_num, height, width, n_channel),
                          dtype=np.uint8)
    y_test_t = np.ndarray((test_t_num, height, width, 1),
                          dtype=np.uint8)
    y_test_f = np.ndarray((test_f_num, height, width, 1),
                          dtype=np.uint8)

    train_t_name = []
    test_t_name = []
    i = 0
    j = 0
    for num, name in enumerate(zip(x_train_t_list, y_train_t_list)):
        x_train = x_train_t_path + name[0]
        y_train = y_train_t_path + name[1]

        if n_channel == 1:
            x_train = image.img_to_array(image.load_img(x_train, grayscale=True, target_size=(height, width)))
        else:
            x_train = image.img_to_array(image.load_img(x_train, grayscale=False, target_size=(height, width)))
        # y_train convert to n_channel = 1 --> grayscale = True
        y_train = image.img_to_array(image.load_img(y_train, grayscale=True, target_size=(height, width)))

        if i != test_t_num:
            if num == random_list_t[i]:
                x_test_t[i] = x_train[:, :, :n_channel]
                y_test_t[i] = y_train
                test_t_name.append(name[0])
                i += 1
            else:
                x_train_t[j] = x_train[:, :, :n_channel]
                y_train_t[j] = y_train
                train_t_name.append(name[0])
                j += 1
        else:
            x_train_t[j] = x_train[:, :, :n_channel]
            y_train_t[j] = y_train
            train_t_name.append(name[0])
            j += 1

    train_f_name = []
    test_f_name = []
    i = 0
    j = 0
    for num, name in enumerate(zip(x_train_f_list, y_train_f_list)):
        x_train = x_train_f_path + name[0]
        y_train = y_train_f_path + name[1]

        if n_channel == 1:
            x_train = image.img_to_array(image.load_img(x_train, grayscale=True, target_size=(height, width)))
        else:
            x_train = image.img_to_array(image.load_img(x_train, grayscale=False, target_size=(height, width)))
        # y_train convert to n_channel = 1 --> grayscale = True
        y_train = image.img_to_array(image.load_img(y_train, grayscale=True, target_size=(height, width)))

        if i != test_f_num:
            if num == random_list_f[i]:
                x_test_f[i] = x_train[:, :, :n_channel]
                y_test_f[i] = y_train
                test_f_name.append(name[0])
                i += 1
            else:
                x_train_f[j] = x_train[:, :, :n_channel]
                y_train_f[j] = y_train
                train_f_name.append(name[0])
                j += 1
        else:
            x_train_f[j] = x_train[:, :, :n_channel]
            y_train_f[j] = y_train
            train_f_name.append(name[0])
            j += 1

    pd.DataFrame(train_t_name, columns=['train_t_name']).to_csv(x_train_t_csv)
    pd.DataFrame(train_f_name, columns=['train_f_name']).to_csv(x_train_f_csv)
    pd.DataFrame(test_t_name, columns=['test_t_name']).to_csv(x_test_t_csv)
    pd.DataFrame(test_f_name, columns=['test_f_name']).to_csv(x_test_f_csv)

    print("test sets have done")
    print("x_train_t_num : {} ，x_train_f_num : {} ".format(x_train_t.shape[0], x_train_f.shape[0]))
    print("x_test_t_num : {} ，x_test_f_num : {} ".format(x_test_t.shape[0], x_test_f.shape[0]))

    for i, data in enumerate(zip(x_train_t, y_train_t)):
        x_train = data[0][np.newaxis, :, :, :]
        y_train = data[1][np.newaxis, :, :, :]

        random_seed = random.randint(0, 9999)
        dataGen(x_train, x_aug_t_path, train_num, random_seed)
        dataGen(y_train, y_aug_t_path, train_num, random_seed)
        if i % 100 == 0:
            print("number {} X_Train_T have done !".format(i))

    for i, data in enumerate(zip(x_train_f, y_train_f)):
        x_train = data[0][np.newaxis, :, :, :]
        y_train = data[1][np.newaxis, :, :, :]

        random_seed = random.randint(0, 9999)
        dataGen(x_train, x_aug_f_path, train_num, random_seed)
        dataGen(y_train, y_aug_f_path, train_num, random_seed)
        if i % 100 == 0:
            print("number {} X_Train_F have done !".format(i))

    for i, data in enumerate(zip(x_test_t, y_test_t)):
        x_test = data[0][np.newaxis, :, :, :]
        y_test = data[1][np.newaxis, :, :, :]

        random_seed = random.randint(0, 9999)
        dataGen(x_test, x_test_t_path, test_num, random_seed)
        dataGen(y_test, y_test_t_path, test_num, random_seed)
        if i % 100 == 0:
            print("number {} X_Test_T have done !".format(i))

    for i, data in enumerate(zip(x_test_f, y_test_f)):
        x_test = data[0][np.newaxis, :, :, :]
        y_test = data[1][np.newaxis, :, :, :]

        random_seed = random.randint(0, 9999)
        dataGen(x_test, x_test_f_path, test_num, random_seed)
        dataGen(y_test, y_test_f_path, test_num, random_seed)
        if i % 100 == 0:
            print("number {} X_Test_F have done !".format(i))

    convertImg(y_aug_t_path)
    convertImg(y_aug_f_path)
    convertImg(y_test_t_path)
    convertImg(y_test_f_path)

    print("aug_generator have done !")

    print("X_Train_T ：{} ，X_Train_F : {} ".format(x_train_t.shape[0], x_train_f.shape[0]))
    print("X_Test_T ：{} ，X_Test_F : {} ".format(x_test_t.shape[0], x_test_f.shape[0]))


def createFile(filepath):
    dir = os.path.split(filepath)[0]
    if not os.path.isdir(dir):
        os.makedirs(dir)


def dataGen(data, file, imgnum, random_seed):
    i = 0
    for _ in data_gen.flow(data, save_to_dir=file,
                           seed=random_seed,
                           save_format=img_type):
        i += 1
        if i >= imgnum:
            break


def convertImg(filepath):
    fileList = os.listdir(filepath)
    for i in fileList:
        path = filepath + i
        data = cv2.imread(path)

        if n_class == 5:
            data[data == 63] = 1
            data[data == 255] = 2
            data[data == 191] = 3
            data[data == 127] = 4
            data[(data != 1) & (data != 2) & (data != 3) & (data != 4)] = 0
        elif n_class == 3:
            data[data == 255] = 2
            data[data == 127] = 1
            data[(data != 0) & (data != 1) & (data != 2)] = 2

        cv2.imwrite(path, data[:, :, 0])


def createNpy(x_t_path=None,
              x_f_path=None,
              y_t_path=None,
              y_f_path=None):
    createFile(npy_path)

    x_train_t_list = os.listdir(x_t_path)
    x_train_f_list = os.listdir(x_f_path)
    y_train_t_list = os.listdir(y_t_path)
    y_train_f_list = os.listdir(y_f_path)

    x_train_t_list.sort()
    x_train_f_list.sort()
    y_train_t_list.sort()
    y_train_f_list.sort()

    t_num = len(x_train_t_list)
    f_num = len(x_train_f_list)

    if x_train_t_list[0] == '.DS_Store':
        x_train_t_list.remove('.DS_Store')
    if y_train_t_list[0] == '.DS_Store':
        y_train_t_list.remove('.DS_Store')

    if len(x_train_t_list) + len(x_train_f_list) != len(y_train_t_list) + len(y_train_f_list):
        print("x_train isn't equal y_train data !")
        return

    x_train_npy = np.ndarray((t_num + f_num, height, width, n_channel),
                             dtype=np.uint8)
    y_train_npy = np.ndarray((t_num + f_num, height * width, n_class),
                             dtype=np.uint8)

    for i, name in enumerate(zip(x_train_t_list, y_train_t_list)):
        x_train = image.img_to_array(image.load_img(x_t_path + name[0], target_size=(height, width)))
        y_train = image.img_to_array(image.load_img(y_t_path + name[1], target_size=(height, width)))

        x_train_npy[i] = x_train[:, :, :n_channel]
        y_train_npy[i] = to_categorical(y_train[:, :, 0].flatten(), n_class)

        if i % 100 == 0:
            print("number {} X_T_Npy have done !".format(i))

    print("T_Npy number : {}".format(t_num))

    for i, name in enumerate(zip(x_train_f_list, y_train_f_list)):
        x_train = image.img_to_array(image.load_img(x_f_path + name[0], target_size=(height, width)))
        y_train = image.img_to_array(image.load_img(y_f_path + name[1], target_size=(height, width)))

        x_train_npy[i + t_num] = x_train[:, :, :n_channel]
        y_train_npy[i + t_num] = to_categorical(y_train[:, :, 0].flatten(), n_class)

        if i % 100 == 0:
            print("number {} X_F_Npy have done !".format(i))

    print("F_Npy number : {}".format(f_num))

    y_train_npy = y_train_npy.reshape(y_train_npy.shape[0], height, width, n_class)

    np.save(x_train_path, x_train_npy)
    np.save(y_train_path, y_train_npy)
    print("X_Train_Shape : ", x_train_npy.shape)
    print("Y_Train_Shape : ", y_train_npy.shape)
    print("data have done !")


class VOCPalette(object):
    def __init__(self, nb_class=21, start=1):
        self.palette = [0] * 768
        # voc2012 21类调色板
        if nb_class > 21 or nb_class < 2:
            nb_class = 21
        if start > 20 or start < 1:
            start = 1
        pal = self.labelcolormap(21)
        self.palette[0] = pal[0][0]
        self.palette[1] = pal[0][1]
        self.palette[2] = pal[0][2]
        for i in range(nb_class):
            self.palette[(i + 1) * 3] = pal[start][0]
            self.palette[(i + 1) * 3 + 1] = pal[start][1]
            self.palette[(i + 1) * 3 + 2] = pal[start][2]
            start = (start + 1) % 21
            if start == 0:
                start = 1
        assert len(self.palette) == 768

    def genlabelpal(self, img_arr):
        img = Image.fromarray(img_arr)
        img.putpalette(self.palette)

        return img

    def genlabelfilepal(self, path, isCoverLab):
        label_list = os.listdir(path)
        label_list.sort()
        for name in label_list:
            if name.endswith(".png"):
                # if name.endswith(".bmp"):
                img = Image.open(path + "/" + name).convert('L')
                shotname, extension = os.path.splitext(name)
                if isCoverLab == True:
                    img_arr = np.array(img)
                    img_arr[np.where(img_arr == 255)] = 1  # for 2 classes: 255->1
                    img = Image.fromarray(img_arr)
                self.palette[0] = 0
                self.palette[1] = 0
                self.palette[2] = 0
                self.palette[14 * 3] = 4
                self.palette[14 * 3 + 1] = 4
                self.palette[14 * 3 + 2] = 4
                self.palette[38 * 3] = 1
                self.palette[38 * 3 + 1] = 1
                self.palette[38 * 3 + 2] = 1
                self.palette[75 * 3] = 2
                self.palette[75 * 3 + 1] = 2
                self.palette[75 * 3 + 2] = 2
                self.palette[113 * 3] = 3
                self.palette[113 * 3 + 1] = 3
                self.palette[113 * 3 + 2] = 3
                img.putpalette(self.palette)
                img.save(path + "/" + shotname + ".png")

    def uint82bin(self, n, count=8):
        """returns the binary of integer n, count refers to amount of bits"""
        return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

    def labelcolormap(self, N):
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r = 0
            g = 0
            b = 0
            id = i
            for j in range(7):
                str_id = self.uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        return cmap


if __name__ == '__main__':
    #调色盘
    # pal =VOCPalette(nb_class=5)
    # pal.genlabelfilepal("../Data/LabelT/",False)
    # pal.genlabelfilepal("../Data/LabelF/",False)

    augGenerator()
    createNpy(x_aug_t_path,x_aug_f_path,y_aug_t_path,y_aug_f_path)

    #Demo:
    # img = cv2.imread(data_path+"YAugT/_0_48.png")
    # print(img)
