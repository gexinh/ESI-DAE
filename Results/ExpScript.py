# -*- coding: utf-8 -*-
"""
2020/5/7
用于测试DAE-mTBF的方法
修改： 通过case来指定具体的数据种类。
pseudo code:
 -construct network
 -load weight
 -load data
  -predict
 -R_2 score
"""
# input: ./py_debug/generate_data/data_for_test/v_1029/case_
# output: ./picture/file_date/

Vert = 1024  # 1024/3002/6004
weight_path = './model_weight' + '.h5'
SNR = 10
# network parameters
time_filter = 3  # time filters
filt = 64  # spatio-temporal feature maps
#
data_name = 'RealData'  # SimData or RealData
"""
#######################################################
                import module
#######################################################
"""
import os
import time
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

# define the name space and get the shape
gain_matrix=gain['Gain_matrix']
if data_name == 'RealData':
    tmc = 48  # time courses
elif data_name == 'SimData':
    tmc = 40
[chan, cortex] = gain_matrix.shape  # channels and vertices
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
        Activation('elu'),
        Conv2D(16, kernel_size=(1, 3), padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               strides=1,
               ),
        BatchNormalization(),
        Activation('elu'),

        Conv2D(filters=24, kernel_size=(1, 5),
               strides=(1, 2), padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        Activation('elu'),
        Conv2D(filters=32, kernel_size=(1, 5),
               strides=1, padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        Activation('elu'),
        Conv2D(filters=36, kernel_size=(1, 5),
               strides=(1, 2), padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        Activation('elu'),

        Conv2D(filters=48, kernel_size=(1, 5),
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
        Activation('elu'),
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
               strides=(2,1), padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        Activation('elu'),
        # part 1
        Conv2D(filters=filt+6, kernel_size=(7, 1),
               strides=1, padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        Activation('elu'),
        Conv2D(filters=filt+12, kernel_size=(7, 1),
               strides=(3,1), padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        Activation('elu'),
        Conv2D(filters=filt+18, kernel_size=(7, 1),
               strides=1, padding='same',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        Activation('elu'),
        Conv2D(filters=filt+24, kernel_size=(17, 1),
               strides=1, padding='valid',
               # activation='relu',
               kernel_initializer='he_normal',
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=l1_l2(l2=1e-4),
               ),
        BatchNormalization(),
        Activation('elu'),
    )
        , name='Encoder'
    )
    Encoder.summary()
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
##################################################################
                    Load weight and Estimate
##################################################################    
"""
if data_name == 'RealData':
    DST_DAE.load_weights(weight_path)
    # to generate a wide range of estimations with making the scale of different digree
    scale = np.linspace(0.1, 2.5, 25)
    real_scalp_ = loadmat('./Test_set/RealData/X_Process.mat')
    real_scalp_ = real_scalp_['X']
    real_scalp = np.tile(np.expand_dims(real_scalp_, axis=0), [scale.size, 1, 1])
    for i in range(len(scale)):
        real_scalp[i, ...] = real_scalp_ * scale[i]

    teNum = 1
    X = Input(shape=(chan, tmc), name='X')
    X_ = Reshape(original_eeg_size)(X)
    Z = Encoder(X_)
    S1 = Spat_Decoder(Z)
    S2 = Temp_Decoder(S1)
    S = S_re_layer(S2)
    Inverse_Model = Model(inputs=X, outputs=S, name='Inverse')
    Inverse_Model.summary()
    # estimate
    start = time.clock()
    S_re_data = Inverse_Model.predict(real_scalp, batch_size=teNum)
    print(time.clock() - start)
    img_path = mkdir('./Test_set/RealData/estimate')
    savemat(img_path + 'source_re.mat', {'source_re': S_re_data})       #S_re
elif data_name == 'SimData':
    pass