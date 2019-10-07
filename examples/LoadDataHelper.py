import os
import sys
import numpy as np
import cv2
import natsort
import logging

class CTADataLoader():
    dataPath = None
    LabelImagesPath = None
    SourceImagesPath = None
    LabelImages = []
    SourceImages = []
    SourceImagesArray = None
    LabelImagesArray = None
    logging.basicConfig(filename="LoadData.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)
    def __init__(self, dataPath):
        # 类的初始化
        self.dataPath = dataPath
        self.LabelImagesPath = dataPath+'/LabelImagesOnlyPlaque/'
        self.SourceImagesPath = dataPath + '/SourceImagesOnlyPlaque/'
        print('Hello' + self.dataPath)
    def loadData(self):
        pass
        self.LabelImages = []
        self.SourceImages = []
        print(self.LabelImagesPath)
        ANumList = os.listdir(self.LabelImagesPath)
        ANumList = natsort.natsorted(ANumList)
        ANum = len(ANumList)#获得A号的个数
        # for i in range(ANum):
        for i in range(1):
            print(str(i)+'/'+str(ANum-1)+'----> Loading ' + ANumList[i])
            logging.info(str(i)+'/'+str(ANum-1)+'----> Loading ' + ANumList[i])
            tmpAPath = self.SourceImagesPath + ANumList[i]
            imagesNameList = os.listdir(tmpAPath)
            imagesNameList = natsort.natsorted(imagesNameList)
            imagesNameNum = len(imagesNameList)
            for j in range(imagesNameNum):
                sourceImageFilePath = self.SourceImagesPath+ ANumList[i]+'/'+imagesNameList[j]
                labelImageFilePath =  self.LabelImagesPath+ ANumList[i]+'/'+imagesNameList[j]
                tmpSourceImage = cv2.imread(sourceImageFilePath, -1)
                if not(tmpSourceImage is None):
                    tmpSourceImage = cv2.resize(tmpSourceImage, (512, 512))
                    self.SourceImages.append(tmpSourceImage)
                else:
                    raise RuntimeError('LoaderError')
                    logging.error('LoaderError tmpSourceImage is None')

                tmpLabelImage = cv2.imread(labelImageFilePath, -1)
                if not (tmpLabelImage is None):
                    tmpLabelImage = cv2.resize(tmpLabelImage, (512, 512))
                    self.LabelImages.append(tmpLabelImage)
                else:
                    raise RuntimeError('LoaderError')
                    logging.error('LoaderError tmpLabelImage is None')
        if(len(self.LabelImages) == len(self.SourceImages)):
            print('I have loaded ' + str(len(self.LabelImages)) + ' Images')
            logging.info('I have loaded ' + str(len(self.LabelImages)) + ' Images')
            #这里步需要将list转换成ndarray, 这里才知道list的长度
            tmpSourceImagesArray = np.zeros((len(self.LabelImages),512, 512,1),dtype = np.float32)
            tmpLabelImagesArray = np.zeros((len(self.LabelImages), 512, 512, 1),dtype = np.float32)
            for ii in range(len(self.LabelImages)):
                tmpSourceImagesArray[ii,::,::,0] = self.SourceImages[ii]/self.SourceImages[ii].max()
                tmpLabelImagesArray[ii, ::, ::, 0] = self.LabelImages[ii]/255
                pass
            self.LabelImagesArray = tmpLabelImagesArray
            self.SourceImagesArray = tmpSourceImagesArray
            return tmpSourceImagesArray, tmpLabelImagesArray
        else:
            raise RuntimeError('LoaderError')
            logging.error('LoaderError len(self.LabelImages) == len(self.SourceImages) not same')
        pass

if __name__ == "__main__":
    print('Hello world')
    myLoader = CTADataLoader('G:/CTAData/ALLDataINLabPCD')
    myLoader.loadData()

