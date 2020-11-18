# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 09:32:03 2019
The training process and estimating phase for DST-DAE
we utilize the training dataset to
@author: Gexin Huang
"""
"""
#######################################
Hyperparameters
#######################################
"""
# data parameters
Vert = 1024  # vertices: 1029\3002\6004
SNR = -5  # estimate the SNR from the real recordings
# the scale of training dataset
sample_num = 2400  # traning samples per batch: 100 300 600 900 1200 1500 1800 2100 2400 2700 3000
split_num = 15    # total samples = sample_num x split_num
# network parameters
time_filter = 3  # time filters
filt = 64  # spatio-temporal feature maps
# loss weight
lambda_1 = 10
lambda_2 = 150
delta = 0.01
# training parameters
nb_epoch = 300
batch_size = 32
lr = 0.001
# training strategy
warm_up_flag = True
dynamic_loss_flag = True
# name
data_name = 'SimData' # SimData or RealData
weight_name = 'DST-DAE'
"""
#######################################################
                import module
#######################################################
"""
import os
# os.environ['THEANO_FLAGS'] = "device=gpu"
os.environ['CUDA_VISIBLE_DEVICES']='0'# CUDA_VISIBLE_DEVICES=1  只有编号为1的GPU对程序是可见的，在代码中gpu[0]对应的就是实际的GPU【1】
#import tool
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from keras.layers import Input, Dense,Flatten, Reshape,Lambda
from keras.layers import BatchNormalization,Activation
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.regularizers import l1, l1_l2, l2
from keras import backend
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint
from keras.callbacks import Callback
import tensorflow as tf
from Noise_Layer import Noise_Layer  # The customized layer which is used to corrupt the signals
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

"""
#####################################################
                Support Function 
#####################################################
"""
####### warm up ################################
class Evaluate(Callback):
    def __init__(self,epoch = 10):
        self.num_passed_batchs = 0
        self.warmup_epochs = epoch       # 确定一下需要多少个epoch进行warmup
    def on_batch_begin(self, batch, logs=None):
        # params是模型自动传递给Callback的一些参数  ： 里面含有各种参数，如 verbosity, batch size, number of epochs...
        # Callback 类里自带了 params 和 model 两个参数，剩下的
        if self.params['steps'] == None:
            self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])  # N/K 的正向取整 = step per epoch
        else:
            self.steps_per_epoch = self.params['steps']
        if self.num_passed_batchs < self.steps_per_epoch * self.warmup_epochs:     #  total step = step per epoch * epoch_num
            # 前10个epoch中，学习率线性地从零增加到0.001
            backend.set_value(self.model.optimizer.lr,
                        lr * (self.num_passed_batchs + 1) / self.steps_per_epoch / self.warmup_epochs)
            self.num_passed_batchs += 1
        # 最后几个epoch应该decay一下
        # if steps_per_epoch * epoch >self.params['epochs']-5
        #
####### callback ###############################
def callbacks(name, path,tensorboard = False,dynamic_loss=True, warm_up = True, epoch =20,):
    callbacks = [
       ModelCheckpoint(path+'weights-{}.h5'.format(str(name)), monitor='val_loss', save_best_only=True, save_weights_only=True)
       #, EarlyStopping(patience=35, monitor='val_loss', min_delta=0, mode='min')
       ,
    ]
    if tensorboard:
        callbacks.append(TensorBoard(log_dir='./logs/{}'.format(name), histogram_freq=0, write_graph=True, write_images=True))
    if warm_up:
        callbacks.append(Evaluate(epoch = epoch))
    if dynamic_loss:
        pass
        #callbacks.append(callback4loss(e1,e2,e3))
    return callbacks
####### make directory ########################
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        # print("---  OK  ---")
        return path + '/'
    else:
        print("---  exist...  ---")
        # print("---  OK  ---")
        return path+'/'

# the save path
# import time
# data = time.localtime()
# day = data[2]
# mon = data[1]
# path = ('./Testing/file_data:%s_%s' % (mon, day))
path = ('./Results/Test_set/'+ data_name )
file_path = mkdir(path)
img_path = file_path + 'estimation'
img_path = mkdir(img_path)
# #########################################################################################################
"""
###############################################
                 load data
 #############################################
"""
if data_name == 'SimData':
    gain = loadmat('./Data Synthesis/support_data/Gain_model_Colin_27_' + str(Vert) + '/Gain_n_' + str(Vert) + '.mat')
elif data_name == 'RealData':
    gain = loadmat('./Data Synthesis/support_data/real_data_' + str(Vert) + '/Gain_n_' + str(Vert) + '.mat')
# source
for i in range(split_num):
    if data_name == 'SimData':
        gen_source = loadmat('./Data Synthesis/training_set/SimData/v_' + str(Vert) + '/source_signal/gen_source_batch_' + str(i + 1) + '.mat')
    elif data_name == 'RealData':
        gen_source = loadmat('./Data Synthesis/training_set/RealData/v_' + str(Vert) + '/source_signal/gen_source_batch_' + str(i + 1) + '.mat')

    if i == 0:
        x = gen_source['gen_tr']
        y = gen_source['gen_te']
        gen_source_tr = x
        gen_source_te = y
    else:
        x = gen_source['gen_tr']
        gen_source_tr = np.concatenate((gen_source_tr,x)) # 3-D tensor 的拼接
        y = gen_source['gen_te']
        gen_source_te = np.concatenate((gen_source_te,y))

# define the name space and get the shape
gain_matrix=gain['Gain_matrix']
gen_source_tr = gen_source_tr[:,:,1:]
gen_source_te = gen_source_te[:,:,1:]
tmc = gen_source_tr.shape[-1]  # time courses
[chan, cortex] = gain_matrix.shape  # channels and vertices
trNum = gen_source_tr.shape[0]
teNum = gen_source_te.shape[0]
# different backend
import sys
script_name = os.path.basename(sys.argv[0]).split(".")[0]
if backend.image_data_format() == 'channels_first':
    original_eeg_size = (1, chan, tmc)
elif backend.image_data_format() == 'channels_last':
    original_eeg_size = (chan, tmc, 1)

"""
#################################################################################
########################## define the neural network model ######################
#################################################################################
"""
#################### define customized function #####################
def forward_model(x, gain=gain_matrix):
    if backend.ndim(x) == 4:
        x = backend.permute_dimensions(x, [0, 3, 2, 1])
        gain = backend.cast(gain, dtype='float32')
        gain = backend.permute_dimensions(gain, [1, 0])
        forward = backend.dot(x, gain)
        forward = backend.permute_dimensions(forward, [0, 3, 2, 1])
    elif backend.ndim(x) == 3:
        x = backend.permute_dimensions(x, [0, 2, 1])
        gain = backend.cast(gain, dtype='float32')
        gain = backend.permute_dimensions(gain, [1, 0])
        forward = backend.dot(x, gain)
        forward = backend.permute_dimensions(forward, [0, 2, 1])
    return forward
def mae_metric(x,xu):
    x = backend.flatten(x)
    xu = backend.flatten(xu)
    return -backend.mean(backend.abs(x-xu))
def mse_metric(x,xu):
    x=backend.flatten(x)
    xu=backend.flatten(xu)
    return -backend.mean((x-xu)**2,axis=-1)
def mse_evaluate(x,x_re):
    mse=np.mean((x-x_re)**2)
    return mse

######################## Input ####################################
S_g = Input(shape=(cortex, tmc), name='gen_source')
Forward_Process = Sequential((
    Lambda(forward_model,
           name='forward_model'
           , trainable=False
           , input_shape=(cortex, tmc)
           ),
)
    , name='Forward_Process'
)
Forward_Process.summary()
######################## Encoder  ####################################
Noise_Block =Sequential((
    Noise_Layer(SNR),
    Reshape((chan, tmc, 1)),
)
    , name='Noise_Block'
)
if data_name == 'SimData':
    Encoder = Sequential((
        Conv2D(8, kernel_size=(1, 3), padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               strides=1,
               name='conv1',
               input_shape=original_eeg_size
               ),
        BatchNormalization(),
        Activation('elu'),
        Conv2D(16, kernel_size=(1, 3), padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               strides=1,
               input_shape=original_eeg_size
               ),
        BatchNormalization(),
        # Activation('relu'),
        Activation('elu'),
        #######################################
        ####### 40 -- 20 ##########
        ######################################
        Conv2D(filters=24, kernel_size=(1, 5),
               strides=(1, 2), padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        # Activation('relu'),
        Activation('elu'),
        # # part 1
        Conv2D(filters=24, kernel_size=(1, 5),
               strides=1, padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        # Activation('relu'),
        Activation('elu'),
        #######################################
        ####### 20 -- 10 ##########
        ######################################

        Conv2D(filters=32, kernel_size=(1, 5),
               strides=(1, 2), padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        # Activation('relu'),
        Activation('elu'),
        # # part 1
        Conv2D(filters=32, kernel_size=(1, 5),
               strides=1, padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        # Activation('relu'),
        Activation('elu'),
        #################################
        ####### 10 -- 5 ##########
        ################################
        Conv2D(filters=64, kernel_size=(1, time_filter),
               strides=(1, 2), padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        # Activation('relu'),
        Activation('elu'),
        # # part 1
        Conv2D(filters=64, kernel_size=(1, time_filter),
               strides=1, padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        # Activation('relu'),
        Activation('elu'),
        #######################################
        ####### 63 -- 21 ##########
        ######################################
        Conv2D(filters=filt, kernel_size=(7, 1),
               strides=(3,1), padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        # Activation('relu'),
        Activation('elu'),
        # # part 1
        Conv2D(filters=filt, kernel_size=(7, 1),
               strides=1, padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        # Activation('relu'),
        Activation('elu'),
        #######################################
        ####### 21 -- 7 ##########
        ######################################
        Conv2D(filters=filt, kernel_size=(7, 1),
               strides=(3,1), padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        # Activation('relu'),
        Activation('elu'),
        # # part 1
        Conv2D(filters=filt, kernel_size=(7, 1),
               strides=1, padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        # Activation('relu'),
        Activation('elu'),
        #######################################
        ####### 7 -- 1 ##########
        ######################################
        Conv2D(filters=filt, kernel_size=(7, 1),
               strides=1, padding='valid',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        # Activation('relu'),
        Activation('elu'),
    )
        , name='Encoder'
    )
elif data_name == 'RealData':
    Encoder = Sequential((
        Conv2D(8, kernel_size=(1, 3), padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               strides=1,
               name='conv1',
               input_shape=original_eeg_size
               ),
        BatchNormalization(),
        # Activation('relu'),
        Activation('elu'),
        # # debug 6/11
        Conv2D(8, kernel_size=(1, 3), padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               strides=1,
               ),
        BatchNormalization(),
        # Activation('relu'),
        Activation('elu'),
        Conv2D(filters=16, kernel_size=(1, 5),
               strides=(1, 2), padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        Activation('elu'),
        Conv2D(filters=16, kernel_size=(1, 5),
               strides=1, padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        Activation('elu'),
        Conv2D(filters=32, kernel_size=(1, 5),
               strides=(1, 2), padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        # Activation('relu'),
        Activation('elu'),

        # debug 6/11
        # part 1
        Conv2D(filters=32, kernel_size=(1, 5),
               strides=1, padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        Activation('elu'),
        Conv2D(filters=64, kernel_size=(1, time_filter),
               strides=(1, 2), padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        # Activation('relu'),
        Activation('elu'),
        # debug 6/11
        # part 1
        Conv2D(filters=64, kernel_size=(1, time_filter),
               strides=1, padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        Activation('elu'),


        Conv2D(filters=filt, kernel_size=(7, 1),
               strides=(2, 1), padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        Activation('elu'),
        # part 1
        Conv2D(filters=filt, kernel_size=(7, 1),
               strides=1, padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        Activation('elu'),
        Conv2D(filters=filt, kernel_size=(7, 1),
               strides=(3, 1), padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        # Activation('relu'),
        Activation('elu'),
        # part 1
        Conv2D(filters=filt, kernel_size=(7, 1),
               strides=1, padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        # Activation('relu'),
        Activation('elu'),

        Conv2D(filters=filt, kernel_size=(17, 1),
               strides=1, padding='valid',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        # Activation('relu'),
        Activation('elu'),
    )
        , name='Encoder'
    )
Encoder.summary()

######################## Decoder  ####################################
# spatial
Spat_Decoder = Sequential((
    Conv2DTranspose(filters=64, kernel_size=(8, 1), strides=(1, 1), padding='valid',
                    # activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=l1_l2(l2=1e-4),
                    bias_initializer='zeros',
                    input_shape=(1, 6, filt)
                    # name='deconv1',
                    ),
    BatchNormalization(),
    Activation('elu'),
    # 8 - 16
    Conv2DTranspose(filters=64, kernel_size=(10, 1), strides=(2, 1), padding='same',
                    # activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=l1_l2(l2=1e-4),
                    bias_initializer='zeros',
                    # name='deconv1',
                    ),
    BatchNormalization(),
    Activation('elu'),
    # 16 - 32
    Conv2DTranspose(filters=32, kernel_size=(10, 1), strides=(2, 1), padding='same',
                    # activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=l1_l2(l2=1e-4),
                    bias_initializer='zeros',
                    # name='deconv1',
                    ),
    BatchNormalization(),
    Activation('elu'),
    # 32 - 64
    Conv2DTranspose(filters=32, kernel_size=(10, 1), strides=(2, 1), padding='same',
                    # activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=l1_l2(l2=1e-4),
                    bias_initializer='zeros',
                    # name='deconv2',
                    ),
    BatchNormalization(),
    Activation('elu'),
    # 64 - 128   # 32 -10
    Conv2DTranspose(filters=32, kernel_size=(10, 1), strides=(2, 1), padding='same',
                    # activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=l1_l2(l2=1e-4),
                    bias_initializer='zeros',
                    # name='deconv1',
                    ),
    BatchNormalization(),
    Activation('elu'),
    # 128 - 256
    Conv2DTranspose(filters=32, kernel_size=(10, 1), strides=(2, 1), padding='same',
                    # activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=l1_l2(l2=1e-4),
                    bias_initializer='zeros',
                    # name='deconv3',
                    ),
    BatchNormalization(),
    Activation('elu'),
    # 256 - 512
    Conv2DTranspose(filters=32, kernel_size=(10, 1), strides=(2, 1), padding='same',
                    # activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=l1_l2(l2=1e-4),
                    bias_initializer='zeros',
                    # name='deconv1',
                    ),
    BatchNormalization(),
    Activation('elu'),
    # 512 - 1024
    Conv2DTranspose(filters=32, kernel_size=(10, 1), strides=(2, 1), padding='same',
                    # activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=l1_l2(l2=1e-4),
                    bias_initializer='zeros',
                    # name='deconv1',
                    ),
    BatchNormalization(),
    Activation('elu'),
)
    , name='Spatial_Deconvolution_Block'
)
Spat_Decoder.summary()
# temporal
Temp_Decoder = Sequential((
    Conv2DTranspose(filters=32,
                    kernel_size=(1, 6),
                    strides=(1, 2),
                    # activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=l1_l2(l2=1e-4),
                    padding='same',
                    # input_shape=(cortex, 5, 16)
                    input_shape=(cortex, 5, 32)
                    ),
    BatchNormalization(),
    # Activation('relu'),
    Activation('elu'),
    # part 1
    Conv2DTranspose(filters=28,
                    kernel_size=(1, 3),
                    strides=1,
                    # activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=l1_l2(l2=1e-4),
                    padding='same',
                    ),
    BatchNormalization(),
    Activation('elu'),
    # 10-20
    Conv2DTranspose(filters=28,
                    kernel_size=(1, 6),
                    strides=(1, 2),
                    # activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=l1_l2(l2=1e-4),
                    padding='same',
                    ),
    BatchNormalization(),
    Activation('elu'),
    # PART 1
    Conv2DTranspose(filters=24,
                    kernel_size=(1, 3),
                    strides=1,
                    # activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=l1_l2(l2=1e-4),
                    padding='same',
                    ),
    BatchNormalization(),
    Activation('elu'),
    ######20 - 40 #############
    # 增加一层解卷积，试试看是否效果会更好？
    Conv2DTranspose(filters=24,
                    kernel_size=(1, 6),
                    strides=(1, 2),
                    # activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=l1_l2(l2=1e-4),
                    padding='same',
                    ),
    BatchNormalization(),
    Activation('elu'),
)
    , name='Temporal_Deconvolution_Block'
)
# last layer: linear activation function
S_re_layer = Sequential((
    Conv2DTranspose(1, kernel_size=(1, 3),
                    kernel_initializer='he_normal',
                    kernel_regularizer=l1(1e-5),
                    strides=1, padding='same'
                    , use_bias=False
                    ),
    BatchNormalization(),

    Reshape((cortex, tmc)),
)
    , name='S_re'
)
Temp_Decoder.summary()

# Forward Transforming Block
Forward_transformer= Sequential((
    Lambda(forward_model,
           name='forward_model'
           , trainable=False
           , input_shape=(cortex, tmc)
           ),
    Reshape((chan, tmc), name='X_re')
)
    , name='Forward_Transformer'
)
Forward_transformer.summary()
###############################################################
"""
##############################################################################################
################################# construct the neural network ###############################
##############################################################################################
"""
X_g = Forward_Process(S_g)
X_n = Noise_Block(X_g)
X_st = Encoder(X_n)
S_t = Spat_Decoder(X_st)
S_re = S_re_layer(Temp_Decoder(S_t))
X_re = Forward_transformer(S_re)
# concatenate
DST_DAE = Model(inputs=S_g, outputs=X_re, name='DST-DAE')
"""
################################################################################################
                                       loss function
################################################################################################
"""
Ls_mse = mse_metric(S_g, S_re)
Ls_mae = mae_metric(S_g, S_re)
Lx_mse = mse_metric(X_g, X_re)
# loss for the scalp and source signals
Lx_ = lambda_1 * Lx_mse
Ls_1_ = lambda_2 * Ls_mse
Ls_2_ = lambda_2 * delta * Ls_mae
# add into the network
DST_DAE.add_loss(Lx_)
DST_DAE.add_loss(Ls_1_)
DST_DAE.add_loss(Ls_2_)
# compile the optimizer
opt_method_bar = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                 decay=0.0)
DST_DAE.compile(optimizer=opt_method_bar)
# add loss
DST_DAE.metrics_tensors.append(Lx_)
DST_DAE.metrics_names.append("Lx_mse")
DST_DAE.metrics_tensors.append(Ls_1_)
DST_DAE.metrics_names.append("Ls_mse")
DST_DAE.metrics_tensors.append(Ls_2_)
DST_DAE.metrics_names.append("Ls_mae")
# summary
DST_DAE.summary()
# print architecture
from keras import utils
utils.plot_model(DST_DAE, to_file='DST-DAE.png')
##########################################################
"""
##################################################################
                        Training process
###################################################################
"""
# Fitting
history = DST_DAE.fit(gen_source_tr,
                                shuffle=True,
                                verbose=2,
                                epochs=nb_epoch,
                                batch_size=batch_size
                                , validation_data=(gen_source_te, None)
                                , callbacks=callbacks(weight_name, img_path, False,dynamic_loss_flag, warm_up_flag)
                        )


"""
##################################################################
                        Testing process
###################################################################
"""
# X_reconstruction
X_re_data = DST_DAE.predict(gen_source_te, batch_size=batch_size)
source_layer = DST_DAE.get_layer('S_re')
source_model = Model(inputs=S_g, outputs=source_layer.get_output_at(1))  ######这里的节点需要随着网络改动
# X_noise
X_n_layer = DST_DAE.get_layer('Noise_Block')
X_n_model = Model(inputs=S_g, outputs=X_n_layer.get_output_at(1))
X_n_data = X_n_model.predict(gen_source_te, batch_size=batch_size)
# X_generate
X_g_layer = DST_DAE.get_layer('Forward_Process')
X_g_model = Model(inputs=S_g, outputs=X_g_layer.get_output_at(1))
X_g_data = X_g_model.predict(gen_source_te, batch_size=batch_size)
# S_reconstruction
S_re_data = source_model.predict(gen_source_te, batch_size=batch_size)

# Metric
from sklearn.metrics import r2_score, mean_squared_error

mse_x = mse_evaluate(X_n_data.flatten(), X_re_data.flatten())
mse_x2 = mse_evaluate(X_g_data.flatten(), X_re_data.flatten())
mse_source = mse_evaluate(gen_source_te.flatten(), S_re_data.flatten())
print('scalp_mse_between_x_n_and_x_re=', mse_x)
print('scalp_mse_between_x_g_and_x_re=', mse_x2)
print('source_mse=', mse_source)

r2_scalp = r2_score(X_n_data.flatten(), X_re_data.flatten())
r2_scalp2 = r2_score(X_g_data.flatten(), X_re_data.flatten())
r2_source = r2_score(gen_source_te.flatten(), S_re_data.flatten())
print('scalp_r2_between_x_n_and_x_re=', r2_scalp)
print('scalp_r2_between_x_g_and_x_re=', r2_scalp2)
print('source_r2=', r2_source)

"""
##################################################################
                        Save estimation
###################################################################
"""
# reconstructed E/MEG signals
x_re = np.transpose(X_re_data, (1, 0, 2))  # 注意matlab和numpy的permute不太一样。
x_re = np.reshape(x_re, [chan, teNum * tmc])
# noised data
x_n = np.transpose(np.reshape(X_n_data, [teNum, chan, tmc]), (1, 0, 2))  # 注意matlab和numpy的permute不太一样。
x_n = np.reshape(x_n, [chan, teNum * tmc])
# simulated data
x_g = np.transpose(np.reshape(X_g_data, [teNum, chan, tmc]), (1, 0, 2))  # 注意matlab和numpy的permute不太一样。
x_g = np.reshape(x_g, [chan, teNum * tmc])
# reconstructed source
s = np.transpose(S_re_data, (1, 0, 2))  # 注意matlab和numpy的permute不太一样。
s = np.reshape(s, [cortex, teNum * tmc])
# simulated source signals from test set.
s_ori = np.transpose(gen_source_te, (1, 0, 2))  # 注意matlab和numpy的permute不太一样。
s_ori = np.reshape(s_ori, [cortex, teNum * tmc])
# save data
savemat(img_path+str(nb_epoch) + '_scalp.mat',{'scalp': X_n_data})
savemat(img_path+str(nb_epoch) + '_scalp_re.mat',{'scalp_re': X_re_data})
savemat(img_path + str(nb_epoch) + '_source_re.mat', {'source_re': S_re_data})
# save weight
DST_DAE.save('./Results/' + 'model_weight.h5')

