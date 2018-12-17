'''
2016 by Jacob Zweig @jacobzweig
build RCNN networks in keras
'''
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import merge, Convolution2D, MaxPooling2D, Input,Add
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
import getData

n_channel = getData.n_channel
height = getData.height
width = getData.width
nbClasses = getData.n_class
nbRCL=5
nbFilters=128
filtersize = 3


def makeModel(n_channel, height, width, nbClasses, nbRCL=5,
		 nbFilters=128, filtersize = 3):


	model = BuildRCNN(n_channel, height, width, nbClasses, nbRCL, nbFilters, filtersize)
	return model

def BuildRCNN(n_channel, height, width, nbClasses, nbRCL, nbFilters, filtersize):
    
    def RCL_block(l_settings, l, pool=True, increase_dim=False):
        input_num_filters = l_settings.output_shape[1]
        if increase_dim:
            out_num_filters = input_num_filters*2
        else:
            out_num_filters = input_num_filters
		   
        conv1 = Convolution2D(nbFilters, (1,1), padding='same', kernel_initializer='he_normal')
        stack1 = conv1(l)   	
        stack2 = BatchNormalization()(stack1)
        stack3 = PReLU()(stack2)
        
        conv2 = Convolution2D(out_num_filters, (filtersize, filtersize), padding='same', kernel_initializer='he_normal')
        stack4 = conv2(stack3)
        stack5 = Add()([stack1,stack4])
        #stack5 = Merge([stack1, stack4], mode='sum')
        stack6 = BatchNormalization()(stack5)
        stack7 = PReLU()(stack6)
    	
        conv3 = Convolution2D(out_num_filters, (filtersize, filtersize), padding='same', kernel_initializer='he_normal')
        stack8 = conv3(stack7)
        stack9 = Add()([stack1,stack8])
        # stack9 = merge([stack1, stack8], mode='sum')
        stack10 = BatchNormalization()(stack9)
        stack11 = PReLU()(stack10)    
        
        conv4 = Convolution2D(out_num_filters, (filtersize, filtersize), padding='same', kernel_initializer='he_normal')
        stack12 = conv4(stack11)
        stack13 = Add()([stack1,stack12])
        # stack13 = merge([stack1, stack12], mode='sum')
        stack14 = BatchNormalization()(stack13)
        stack15 = PReLU()(stack14)    
        
        if pool:
            stack16 = MaxPooling2D(pool_size=(2, 2),padding='same')(stack15)
            stack17 = Dropout(0.1)(stack16)
        else:
            stack17 = Dropout(0.1)(stack15)
            
        return stack17

    #Build Network
    input_img = Input(shape=(height, width, n_channel))
    conv_l = Convolution2D(nbFilters, (filtersize,filtersize), activation='relu',padding='same', kernel_initializer='he_normal')
    l = conv_l(input_img)
    
    for n in range(nbRCL):
        if n % 2 ==0:
            l = RCL_block(conv_l, l, pool=False)
        else:
            l = RCL_block(conv_l, l, pool=True)
    
    out = Flatten()(l)        
    l_out = Dense(nbClasses, activation = 'softmax')(out)
    
    model = Model(input = input_img, output = l_out)
    
    return model

'''
# Keras-RCNN

Some experimenting with Keras to build Recurrent Convolutional Neural Networks, based on the paper [Recurrent Convolutional Neural Network for Object Recognition](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liang_Recurrent_Convolutional_Neural_2015_CVPR_paper.pdf). 

```
# Build a model
model = BuildRCNN(n_channel, height, width, nbClasses, nbRCL, nbFilters, filtersize)
_where_
...n_channel -> number of channels
...height, width -> dimensions of image
...nbClasses -> number of classes
...nbRCL -> number of RCL block (default = 5). Defines the depth of recurrence
...nbFilters -> number of filters
...filtersize -> size of the filter

#Compile it
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#fit
model.fit(X_train, y_train, batch_size=64, nb_epoch=100, validation_data = (X_valid, y_valid))
```
'''
def getModel():
    def RCL_block(l_settings, l, pool=True, increase_dim=False):
        input_num_filters = l_settings.output_shape[3]
        print("input_num_filters:",input_num_filters)
        if increase_dim:
            out_num_filters = input_num_filters*2
        else:
            out_num_filters = input_num_filters

        conv1 = Convolution2D(nbFilters, (1,1), padding='same', kernel_initializer='he_normal')
        stack1 = conv1(l)
        stack2 = BatchNormalization()(stack1)
        stack3 = PReLU()(stack2)

        conv2 = Convolution2D(out_num_filters, (filtersize, filtersize), padding='same', kernel_initializer='he_normal')
        stack4 = conv2(stack3)
        stack5 = Add()([stack1,stack4])
        #stack5 = Merge([stack1, stack4], mode='sum')
        stack6 = BatchNormalization()(stack5)
        stack7 = PReLU()(stack6)

        conv3 = Convolution2D(out_num_filters, (filtersize, filtersize), padding='same', kernel_initializer='he_normal')
        stack8 = conv3(stack7)
        stack9 = Add()([stack1,stack8])
        # stack9 = merge([stack1, stack8], mode='sum')
        stack10 = BatchNormalization()(stack9)
        stack11 = PReLU()(stack10)

        conv4 = Convolution2D(out_num_filters, (filtersize, filtersize), padding='same', kernel_initializer='he_normal')
        stack12 = conv4(stack11)
        stack13 = Add()([stack1,stack12])
        # stack13 = merge([stack1, stack12], mode='sum')
        stack14 = BatchNormalization()(stack13)
        stack15 = PReLU()(stack14)

        if pool:
            stack16 = MaxPooling2D(pool_size=(2, 2),padding='same')(stack15)
            stack17 = Dropout(0.1)(stack16)
        else:
            stack17 = Dropout(0.1)(stack15)

        return stack17

    #Build Network
    input_img = Input(shape=(height, width, n_channel))
    conv_l = Convolution2D(nbFilters, (filtersize,filtersize), activation='relu',padding='same', kernel_initializer='he_normal')
    l = conv_l(input_img)

    for n in range(nbRCL):
        if n % 2 ==0:
            l = RCL_block(conv_l, l, pool=False)
        else:
            l = RCL_block(conv_l, l, pool=True)

    print("l_shape:",l)

    out = Flatten()(l)
    l_out = Dense(nbClasses, activation = 'softmax')(out)

    model = Model(input = input_img, output = l_out)

    return model

