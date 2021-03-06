# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 19:12:53 2017

@author: Weidi Xie

@Description: This is the file used for training, loading images, annotation, training with model.
"""

import numpy as np
import pdb                                 #调试模块
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import matplotlib.pyplot as plt
from generator import ImageDataGenerator
from model import buildModel_U_net
from keras import backend as K
from keras.callbacks import ModelCheckpoint,Callback,LearningRateScheduler
from scipy import misc                   # miscellaneous 杂项
import scipy.ndimage as ndimage

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

base_path = 'cells/'
data = []
anno = []

def step_decay(epoch):
    step = 16
    num =  epoch // step 
    if num % 3 == 0:
        lrate = 1e-3
    elif num % 3 == 1:
        lrate = 1e-4
    else:
        lrate = 1e-5
        #lrate = initial_lrate * 1/(1 + decay * (epoch - num * step))
    print('Learning rate for epoch {} is {}.'.format(epoch+1, lrate))
    return np.float(lrate)

#---- 读取文件数据   
def read_data(base_path):
    imList = os.listdir(base_path)  # 获取data文件夹下面的文件名称列表
    for i in range(len(imList)):     
        if 'cell' in imList[i]:
            img1 = misc.imread(os.path.join(base_path,imList[i]))# 读取cell.png文件中的数据
            data.append(img1)
            
            img2_ = misc.imread(os.path.join(base_path, imList[i][:3] + 'dots.png'))# 读取dots.png文件中的数据
            img2 = 100.0 * (img2_[:,:,0] > 0) # 找到像素值大于0的 marsk
            img2 = ndimage.gaussian_filter(img2, sigma=(1, 1), order=0) # 对mask进行高斯滤波
            anno.append(img2)
    return np.asarray(data, dtype = 'float32'), np.asarray(anno, dtype = 'float32') # 将data和anno转换成数组
    
def train_(base_path):
    data, anno = read_data(base_path) # type: float32
    anno = np.expand_dims(anno, axis = -1) # 扩展数组的形状
    
    mean = np.mean(data)
    std = np.std(data)
    
    data_ = (data - mean) / std
    
    train_data = data_[:150]
    train_anno = anno[:150]

    val_data = data_[150:]
    val_anno = anno[150:]
    
    print('-'*30)  
    print('Creating and compiling the fully convolutional regression networks.')
    print('-'*30)    
   
    model = buildModel_U_net(input_dim = (256,256,3))
    model_checkpoint = ModelCheckpoint('cell_counting.hdf5', monitor='loss', save_best_only=True)
    model.summary()
    print('...Fitting model...')
    print('-'*30)
    change_lr = LearningRateScheduler(step_decay)

    datagen = ImageDataGenerator(
        featurewise_center = False,  # set input mean to 0 over the dataset
        samplewise_center = False,  # set each sample mean to 0
        featurewise_std_normalization = False,  # divide inputs by std of the dataset
        samplewise_std_normalization = False,  # divide each input by its std
        zca_whitening = False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range = 0.3,  # randomly shift images horizontally (fraction of total width)
        height_shift_range = 0.3,  # randomly shift images vertically (fraction of total height)
        zoom_range = 0.3,
        shear_range = 0.,
        horizontal_flip = True,  # randomly flip images
        vertical_flip = True, # randomly flip images
        fill_mode = 'constant',
        dim_ordering = 'tf')  

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(train_data,
                                     train_anno,
                                     batch_size = 16
                                     ),
                        samples_per_epoch = train_data.shape[0],
                        nb_epoch = 192,
                        callbacks = [model_checkpoint, change_lr],
                       )
    
    model.load_weights('cell_counting.hdf5')
    A = model.predict(val_data)
    mean_diff = np.average(np.abs(np.sum(np.sum(A,1),1)-np.sum(np.sum(val_anno,1),1))) / (100.0)
    print('After training, the difference is : {} cells per image.'.format(np.abs(mean_diff)))
    
if __name__ == '__main__':
    train_(base_path)
    print("management the project using Git")
