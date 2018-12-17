from keras import Model
import numpy as np

import FuzzyUnet
import getData

model_path = "../Model/fuzzy_unet_batchsize_10_epochs_20_1114_18_39.h5"

def loadModel():
    x_train = np.load(getData.x_train_path).astype('float32')
    y_train = np.load(getData.y_train_path).astype('float32')

    # 0~1归一化
    x_train /= 255

    model = FuzzyUnet.getModel()

    model.load_weights(model_path)

    layer_model = Model(inputs=model.input,outputs=model.get_layer('activation_1').output)

    output = layer_model.predict(x_train)

    print(output)

    print("output_shape",output.shape)
    print("output:",output[0])

loadModel()





