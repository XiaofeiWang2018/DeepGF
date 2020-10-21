import numpy as np
import sys
import os
from data_processing import DataLoader_atten_polar
""" num of 0/1 in prediction"""
# list_img_path_train_init = os.listdir('./data/test/image/all')
# list_img_path_train_init.sort()
# dataloader_train = DataLoader_atten_polar(batch_size=4, list_img_path=list_img_path_train_init, state='train')
# p=0
# n=0
# for batch in range(int(384/4)):
#     imagev, yearv, GTmapv, Polarv, GTlabelv = dataloader_train.get_batch()
#     GTlabelv_test = np.reshape(GTlabelv, [-1])
#     for nb in range(4 * 5):
#         if GTlabelv_test[nb] == 1:
#             p = p + 1
#         if GTlabelv_test[nb] == 0 :
#             n = n + 1


list_img_path_train_init = os.listdir('./data/test/image/all')
list_img_path_train_init.sort()
list_img_path_train_aug=[]
k_0=0
k_1=0
k_2=0
k_3=0
k_4=0
k_5=0
for i in range(len(list_img_path_train_init)): #348
    label = 0
    list_img_path_train_aug.append(list_img_path_train_init[i])
    with open('./data/test/label/all/' + list_img_path_train_init[i] + '.txt', 'r') as f:
        K = f.readlines()
        for i_line in range(5):
            line = K[i_line + 1]
            line = line.strip('\n')
            line = int(line)
            label +=line
    if label == 0 :
        k_0+=1
    if label == 1 :
        k_1+=1
    if label == 2 :
        k_2+=1
    if label == 3 :
        k_3+=1
    if label == 4 :
        k_4+=1
    if label == 5 :
        k_5+=1
    # if label >= 3:
    #     for j in range(15):
    #         list_img_path_train_aug.append(list_img_path_train_init[i])
list_img_path_train=list_img_path_train_aug

# dataloader_train = DataLoader_atten_polar(batch_size=4, list_img_path=list_img_path_train, state='train')
# p=0
# n=0
# for batch in range(int(len(list_img_path_train)/4)):
#     GTlabelv = dataloader_train.get_batch_1()
#     GTlabelv_test = np.reshape(GTlabelv, [-1])
#     for nb in range(4 * 5):
#         if GTlabelv_test[nb] == 1:
#             p = p + 1
#         if GTlabelv_test[nb] == 0 :
#             n = n + 1
#     print(batch)
#
# a=1
#
#
#
# import tensorflow as tf
# import random
# import CNN_ALEX as Network
# import time
# import math
# import os
# from matplotlib import pyplot as plt
# from data_processing import DataLoader_atten_polar
#
#
# list_img_path_train_init = os.listdir('./data/train/image/all')
# list_img_path_train_init.sort()
# list_img_path_train=list_img_path_train_init
# list_img_path_test = os.listdir('./data/test/image/all')
# list_img_path_test.sort()
# """train"""
# dataloader_train = DataLoader_atten_polar(batch_size=1308, list_img_path= list_img_path_train,state='train')
# """test"""
# dataloader_test = DataLoader_atten_polar(batch_size=348,list_img_path=list_img_path_test, state='test')
#
#
#
# year1= dataloader_test.get_batch_1()
# a1=1















