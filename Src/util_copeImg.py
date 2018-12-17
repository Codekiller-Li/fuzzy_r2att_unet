# coding = utf-8
import numpy as np
import cv2
import PIL
import Unet
import getData

imgfile = "../Img/20150305141007348.jpeg"
img = cv2.imread(imgfile)

test_num=1


def util_copeImg(imgData=None):
    imgData = imgData.astype(np.uint8)

    copy_img = imgData.copy()
    copy_img = copy_img.reshape((copy_img.shape[0], copy_img.shape[1], copy_img.shape[2]))
    copy_img[copy_img == 1] = 255
    copy_img[copy_img != 255] = 0

    for num, img in enumerate(copy_img):
#        cv2.imwrite("../Img/{}_ori.jpg".format(num), imgData[num])
        _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        c_max = []
        max_area = 0
        max_cnt = 0
        print("length:",len(contours))
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            # find max countour
            if (area > max_area):
                if (max_area != 0):
                    c_min = []
                    c_min.append(max_cnt)
                    cv2.drawContours(imgData[num], c_min, -1, (2, 2, 2), cv2.FILLED)
                max_area = area
                max_cnt = cnt
            else:
                c_min = []
                c_min.append(cnt)
                cv2.drawContours(imgData[num], c_min, -1, (2, 2, 2), cv2.FILLED)

        c_max.append(max_cnt)

        cv2.drawContours(imgData[num], c_max, -1, (1, 1, 1), thickness=-1)
#        cv2.imwrite("../Img/{}_cope.jpg".format(num), imgData[num])

    for i in range(test_num):
        img_ori = cv2.imread("../Img/{}_ori.jpg".format(i))
        img_cope = cv2.imread("../Img/{}_cope.jpg".format(i))

        img_ori[img_ori == 1] = 255
        img_ori[img_ori == 2] = 127

        img_cope[img_cope == 1] = 255
        img_cope[img_cope == 2] = 127

        cv2.imwrite("../Img/{}_ori_af.jpg".format(i), img_ori)
        cv2.imwrite("../Img/{}_cope_af.jpg".format(i), img_cope)
    return imgData


model_path = "../Model/unet_batchsize_10_epochs_100_1111_10_32.h5"

if __name__ == "__main__":
    x_train = np.load("../NpyData/train.npy").astype('float32')
    y_train = np.load("../NpyData/label.npy").astype('float32')

    x_train = x_train[:test_num, :, :, :]
    y_train = y_train[:test_num, :, :, :]

    # print("x_train",y_train[:,:,:,1])
    model = Unet.getModel()
    model.load_weights(model_path)
    y_predict = model.predict(x_train, batch_size=2, verbose=0)

    y_predict = y_predict.reshape(y_train.shape[0], getData.height * getData.width, getData.n_class)

    y_predict = np.argmax(y_predict, axis=2)

    y_predict = y_predict.reshape(y_predict.shape[0], getData.height, getData.width, 1)

    util_copeImg(y_predict)

    # img = cv2.imread("../Img/20150305141007348.jpeg")
    # h, w, _ = img.shape
    #
    # print("shape_1:",img.shape)
    #
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    #
    # print("shape_2",thresh.shape)
    #
    # _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #
    # c_max = []
    # max_area = 0
    # max_cnt = 0
    # for i in range(len(contours)):
    #     cnt = contours[i]
    #     area = cv2.contourArea(cnt)
    #     # find max countour
    #     if (area > max_area):
    #         if (max_area != 0):
    #             c_min = []
    #             c_min.append(max_cnt)
    #             cv2.drawContours(img, c_min, -1, (0, 0, 0), cv2.FILLED)
    #         max_area = area
    #         max_cnt = cnt
    #     else:
    #         c_min = []
    #         c_min.append(cnt)
    #         cv2.drawContours(img, c_min, -1, (0, 0, 0), cv2.FILLED)
    #
    # c_max.append(max_cnt)
    #
    # cv2.drawContours(img, c_max, -1, (255, 255, 255), thickness=-1)
    #
    # cv2.imwrite("../Img/test.jpg",img)
