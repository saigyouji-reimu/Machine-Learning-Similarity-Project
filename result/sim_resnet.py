from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import os
from scipy import io

# parameters for computing the similarity
num1 = 4
num2 = 6
if_relu = 0       #if get through the relu activation

if if_relu:
    layer_name = 'activation_'
    layer_num = 19
else:
    layer_name = 'conv2d_'
    layer_num = 21
# end parameters

# the model
n = 3
version = 1
num_classes = 10
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

input_shape = x_train.shape[1:]

x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer=tf.keras.initializers.RandomNormal(),
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
# end model

#load the trained models
model1 = resnet_v1(input_shape=input_shape, depth=depth)
model2 = resnet_v1(input_shape=input_shape, depth=depth)
s = './saved_models/ResNet20_'
model1.load_weights(s+str(num1)+'.h5')
model2.load_weights(s+str(num2)+'.h5')
model1.summary()
model2.summary()

#compute each layer's output
list_conv = []
all=7
for i in range(layer_num-all,layer_num):
    if i!=0:
        s = layer_name + str(i)
    else:
	    s = layer_name[0:-1]
	
    temp = Model(inputs=model1.input, outputs=model1.get_layer(s).output).predict(x_test).reshape((10000,-1))
    temp_mean = np.sum(temp,axis=0)/10000
    temp = temp - temp_mean
    temp = temp.transpose()
    list_conv.append(temp)

for i in range(layer_num-all,layer_num):
    s = layer_name + str(i+layer_num)
    temp = Model(inputs=model2.input, outputs=model2.get_layer(s).output).predict(x_test).reshape((10000,-1))
    temp_mean = np.sum(temp,axis=0)/10000
    temp = temp - temp_mean
    temp = temp.transpose()
    list_conv.append(temp)

#the linear CKA
def CKA(x,y):
    a = norm(y.transpose().dot(x))
    b = norm(x.transpose().dot(x))
    c = norm(y.transpose().dot(y))
    return (a*a) / (b*c)

#matlab
def OMMD(y, z):
    io.savemat(r"Y.mat", {'data': y})
    io.savemat(r"Z.mat", {'data': z})
    os.system("matlab -nodesktop -nosplash -r test")
    d = io.loadmat(r"DIST.mat")
    print('DIST calculated')
    print(d['DIST'][0][0])
    return d['DIST'][0][0]

#compute the similarity
list_sim = []
for i in range(all):
    print("compute:", i)
    for j in range(all):
        if i <= j:
            list_sim.append(OMMD(list_conv[i], list_conv[j+all]))
        else:
            list_sim.append(list_sim[all*j+i])

#visualize
list_sim = np.array(list_sim).reshape(all,all)
print(list_sim)
np.save("sim_resnet.npy",list_sim)
plt.imshow(list_sim)
plt.colorbar(shrink=.92)

plt.xticks(np.arange(0,all))
plt.yticks(np.arange(0,all))
plt.show()
