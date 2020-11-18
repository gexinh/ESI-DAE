# -*- coding: utf-8 -*-
"""
the script to implement data synthesis strategy for DST-DAE
@author: Gexin Huang
"""
# =============================================================================
# import module and data
# =============================================================================
from scipy.io import loadmat, savemat
import numpy as np
import matlab.engine
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os
import random
################# flag for the simulation or real experiments ##################
TBF_flag = 1   # 0: simulate TBF ; 1: real TBF
# =============================================================================
# import data
# =============================================================================
eng = matlab.engine.start_matlab()

if TBF_flag == 0:
    TBF = loadmat('./support_data/TBF_Sim.mat')
	path_flag = 0
    vert = loadmat('./support_data/Gain_model_Colin_27_'+str(vert_mode)+'/VertC')
elif TBF_flag == 1:
    TBF = loadmat('./support_data/TBF_real.mat')
	path_flag = 1
	vert = loadmat('./support_data/real_data_'+str(vert_mode)+ '/VertC')
	
TBF = TBF['TBF']

VertC = vert['VertConn']
cort = VertC.shape[0]
tmc = TBF.shape[-1]
# =============================================================================
# parameters
# =============================================================================

# =============================================================================
case = 4  # 1,2: single; 3: double: 4: triple 5: four
# =============================================================================
sample_num_ = 2400      # samples at each batch: 100 300 600 900 1200 1500 1800 2100 2400 2700 3000   #最大是 2**32 byte =4 G    1 G =2**30   1MB =2**10 byte
batch_num= 15           # batch numbers
vert_mode = 1024 # 1024 3002 6004 

count_ = 0  # 用来确定当前的batch是training set or test set
split_rate = 0.8 # cross validation rate
sample_num = sample_num_
##################  setting 4 case for generate #########################
if case==1:   # 单源，但是尺寸不同
    generate_mode=2
    SourceNum = 1
elif case==2:   # 时间信号相同的双源
    generate_mode=3
    SourceNum =2
elif case==3:   # 时间信号不同的三源
    generate_mode = 4
    SourceNum = 3
elif case == 4:
    generate_mode = 5
    SourceNum = 4
# =============================================================================
# support function
# =============================================================================
def Add_noise(x, d, SNR):
    P_signal = np.sum(abs(x) ** 2)
    P_d = np.sum(abs(d) ** 2)
    P_noise = P_signal / 10 ** (SNR / 10)
    noise = np.sqrt(P_noise / P_d) * d
    noise_signal = x + noise
    return noise_signal
def wgn(x, snr):
    P_signal = np.var(x, axis=-1)
    P_noise = P_signal / (10 ** (snr / 10.0))
    noise = np.random.randn(x.shape[-2], x.shape[-1])
    return x + noise * np.expand_dims(np.sqrt(P_noise), -1)
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


    """
        获得m个长度为n的随机向量
        随机向量元素的值在[a, b]范围内
        随机向量元素的和为s
        return 为[n, m]矩阵
                np.sum(x, 0)为[1, 1, ..]
    """
    assert n * a <= s <= n * b and a < b
    n, m = int(n), int(m)
    s = (s - n * a) / (b - a)
    k = int(max(min(np.floor(s), n - 1), 0))

    s1 = s - np.arange(k, k - n, -1)
    s2 = np.arange(k + n, k, -1) - s

    w = np.zeros((n, n + 1))
    w[0, 1] = np.inf

    t = np.zeros((n - 1, n))
    tiny = 1e-32

    for i in range(2, n + 1):
        tmp1 = w[i - 2, 1:i + 1] * s1[0:i] / i
        tmp2 = w[i - 2, 0:i] * s2[n - i: n] / i
        w[i - 1, 1:i + 1] = tmp1 + tmp2
        tmp3 = w[i - 1, 1:i + 1] + tiny
        tmp4 = (s2[n - i: n] > s1[0: i]).astype(np.float)
        t[i - 2, 0:i] = (tmp2 / tmp3) * tmp4 + (1 - tmp1 / tmp3) * (1 - tmp4)

    x = np.zeros((n, m))
    if m == 0:
        return x

    rt = np.random.rand(n - 1, m)
    rs = np.random.rand(n - 1, m)

    s = s * np.ones((1, m)).astype(np.int)
    j = (k + 1) * np.ones((1, m)).astype(np.int)
    sm = np.zeros((1, m))
    pr = np.ones((1, m))

    for i in range(n - 1, 0, -1):
        e = rt[n - i - 1, :] <= t[i - 1, j - 1]
        sx = rs[n - i - 1, :] ** (1 / i)
        sm += (1 - sx) * pr * s / (i + 1)
        pr = sx * pr
        x[n - i - 1, :] = sm + pr * e
        s = s - e
        j = j - e

    x[n - 1, :] = sm + pr * s
    x = (b - a) * x + a
    return x

# =============================================================================
# data synthesis
# =============================================================================
for j in range(batch_num):
    while True:
        sNum = sample_num
        generate_source = np.ndarray(shape=(sNum, cort, tmc))
        activate_area = np.ndarray(shape=(sNum, cort, SourceNum))

        # =============================================================================
        #  call the .m file 
        # =============================================================================
		area =  spat_gen(sample_num, SourceNum, vert_mode, generate_mode, path_flag, (2 * (j + 1) - 1) + count_)

        # =============================================================================
        # load cell file
        # =============================================================================
        dic=loadmat('./generate_data/area_dic_'+str(2*(j+1)-1+count_)+'_.mat')
        # dic = loadmat('./generate_data/area_dic.mat')
        dic = dic['dic_'+str(2*(j+1)-1+count_)]

        lamb_d = loadmat('./generate_data/area_lamb_' + str(2 * (j + 1) - 1 + count_) + '_.mat')
        lamb_d = lamb_d['lamb_' + str(2 * (j + 1) - 1 + count_)]
		
        # =============================================================================
        # data synthesis from spatial and temporal componets.
        # =============================================================================
        for sample in range(sample_num):
            source_1 = np.zeros((cort, tmc))          #时间信号
            source_2 = np.zeros((cort, SourceNum))    #激活标签
			
            for i in range(SourceNum):
                dic_ = dic[sample][i]
                dic_ = np.squeeze(dic_)  # 缩减一个维度
                source_size = np.size(dic_)
                lamb = lamb_d[sample][i]
				source_1[dic_ - 1, :] = np.tile(np.dot(lamb, TBF), (source_size, 1))
                source_2[dic_ - 1, i ] = 1
            ##########################################################################
            source_1 = source_1.reshape(1, cort, tmc)
            source_2 = source_2.reshape(1, cort, SourceNum)

            generate_source[sample, :, :] = source_1
            activate_area[sample, :, :] = source_2

    # =============================================================================
    #  split and save the dataset
    # =============================================================================
      
      # ===================    shuffle  =====================================
        if count_ == 0: # count_ = 0 :trainning set ; count_ = 1 :testing set
            x, y = shuffle(generate_source, activate_area)
            sample_num = int(np.round(sample_num_ * simalar_sample_ * (1 - split_rate)))  # .astype(np.int) 这是32
        elif count_ == 1:
            x_t, y_t = shuffle(generate_source, activate_area)
            sample_num = sample_num_
            count_ = 0
            break
        count_ += 1 
    
    # ======================  save   ===================================
    # name: gen_source_batch_xx.mat ; gen_source_batch_xx_label.mat
    # content: gen_tr: trainning set ; gen_te: testing set
	
	if TBF_flag = 0
		 path = mkdir('./training_set/Sim_' + '/v_' + str(vert_mode))
	else
		 path = mkdir('./training_set/Real_' + '/v_' + str(vert_mode))
		
    path = mkdir(path+'/sample_' + str(sample_num_ * batch_num))
    path_x = mkdir(path+'source_signal')
    savemat(path_x + 'gen_source_batch_' + str(j + 1) + '.mat',{ 'gen_tr': x, 'gen_te': x_t})
    path_y = mkdir(path+'source_label')
    savemat(path_y + 'gen_source_batch_' + str(j + 1) + '_label.mat',{ 'gen_tr': y, 'gen_te': y_t})