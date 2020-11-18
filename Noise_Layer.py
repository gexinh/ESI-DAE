# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:55:30 2020
add batch noise 

@author: Bigyee
"""
from keras import backend as K
from keras.engine.topology import Layer
from keras.utils import conv_utils
from keras import initializers
from keras import activations
from keras import regularizers
from keras import constraints
import numpy as np

class Noise_Layer(Layer):

    def __init__(self,
                 SNR,
                 **kwargs):
        super(Noise_Layer, self).__init__(**kwargs)      #这句话还是很关键的
        ### 定义参数 ###
        self.snr= SNR
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    def call(self, inputs):
        ## define the function, 用函数包装，可以保证每次调用时噪音不一样 :
        def noised(x,snr):
            noise=K.random_normal(shape=(K.shape(x)),mean=0.,stddev=1.0)  #要去broadcast 到样本维度
            ## 均值是0就不需要再去了 ##
#            noise_u=K.tile(K.mean(noise,axis=-1,keepdims=True) #均值化之后为(batch,c,1)的向量
#                                ,(1,1,self.t))
#            noise=noise-noise_u   #去均值化  :(batch,c,t)
            signal_power=K.var(x,axis=-1,keepdims=False)   #信号方差：(batch,c)
            noise_var =signal_power/(10**(snr/10))    #sample*chan     :(batch,c)
            amplitude=K.sqrt(noise_var)/K.std(noise,axis=-1,keepdims=False)    #(batch,c)
            amplitude = K.expand_dims(amplitude,axis=-1)   #aplitude_size (batch,c,1)
            noise = amplitude * noise                      #再一次广播
            output = x + noise 
#            output = K.expand_dims(signal_power,axis=-1)
            return output
        return noised(inputs,self.snr)

    def compute_output_shape(self, input_shape):
        #用来防止维度出错时，也能知道输出的维度
        # define the output shape ,and let the keras know hot to derivate the new layers
        return input_shape