from matplotlib import pyplot as plt
import numpy as np
import h5py
import os

HalfLen = 5 # LSTM中输入的序列长度一半


train_path = ['Training.mat']
# train_path.append('J:/头颈血管重建VNET/CTA血管自动分割VNET/原始数据/5/分割V/DataNormalized.mat')
# train_path.append('J:/头颈血管重建VNET/CTA血管自动分割VNET/原始数据/6/分割V/DataNormalized.mat')
# train_path.append('J:/头颈血管重建VNET/CTA血管自动分割VNET/原始数据/7/分割V/DataNormalized.mat')

test_path = ['Training.mat']
# test_path = ['J:/头颈血管重建VNET/CTA血管自动分割VNET/原始数据/3/分割V/DataNormalized.mat']
# test_path.append('J:/头颈血管重建VNET/CTA血管自动分割VNET/原始数据/7/分割V/DataNormalized.mat')


def create_train_data():
    imgs_train = []
    labs_train = []
    for cont in range(len(train_path)):
        # 遍历训练文件
        data = h5py.File(train_path[cont])
        TraIMG = np.transpose(data['TraIMG'])
        TraMSK = np.transpose(data['TraMSK'])
        # PCHRCS_Vslness = np.ndarray((image_all.shape[0],image_all.shape[1],:), dtype=np.float32)
        
        imgs_train = TraIMG[...,np.newaxis]
        labs_train = TraMSK[...,np.newaxis]
    
    np.save('imgs_train.npy', imgs_train)
    np.save('labs_train.npy', labs_train)
    
    
def create_test_data():
    imgs_test = []
    labs_test = []
    for cont in range(len(test_path)):
        # 遍历训练文件
        data = h5py.File(train_path[cont])
        TstIMG = np.transpose(data['TstIMG'])
        TstMSK = np.transpose(data['TstMSK'])

        imgs_test = TstIMG[...,np.newaxis]
        labs_test = TstMSK[...,np.newaxis]
    
    np.save('imgs_test.npy', imgs_test)
    np.save('labs_test.npy', labs_test)
    
    
def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    labs_train = np.load('labs_train.npy')
    return imgs_train, labs_train
    
    
def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    labs_test = np.load('labs_test.npy')
    return imgs_test, labs_test
    

    
                        
            
            
            
            
            
        
    
    
    




# imgs_d = np.transpose(data['PCHTRAIN_Dark'])
# imgs_v = np.transpose(data['PCHTRAIN_Vessel'])
    

