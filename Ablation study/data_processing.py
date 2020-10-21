
import tensorflow as tf
import numpy as np
import random
from PIL import Image
from matplotlib import pyplot as plt
import glob
from skimage import io,transform
import os
import scipy.io as scio



class DataLoader_atten_polar(object):

    def __init__(self, batch_size, list_img_path,state='train'):
        # reading data list

        self.state=state
        random.seed(20190510)
        self.path_to_image = '../N19_our_model/data/'+self.state+'/image/all/'
        self.path_to_label = '../N19_our_model/data/' + self.state + '/label/all/'
        self.path_to_atten = '../N19_our_model/data/final_atten/'
        self.path_to_polar = '../N19_our_model/data/final_polar/'
        self.batch_size = batch_size
        self.list_img_path = list_img_path
        self.size = len(self.list_img_path)
        self.num_batches = int(self.size / self.batch_size)
        self.cursor = 0
        self.batch_order=0

    def get_batch(self):  # Returns
        if self.cursor + self.batch_size > self.size:
            self.cursor = 0
            self.batch_order = 0
            np.random.shuffle(self.list_img_path)
        img_batch = []
        year_batch = []
        Atten_map_batch = []
        Polar_map_batch = []
        label_batch=[]
        for idx in range(self.batch_size):
            img = []
            year = []
            label=[]
            Atten_map = []
            Polar_map = []
            count = 0
            image_subpath=self.list_img_path[self.batch_order * 4 + idx]
            with open(self.path_to_label+image_subpath+'.txt', 'r') as f:
                K = f.readlines()
                for i_line in range(5):
                    line= K[i_line+1]
                    line = line.strip('\n')
                    line = int(line)
                    label.append(line)
            image_sublist = glob.glob(self.path_to_image + '/' + image_subpath+ '/' + '*.jpg')
            image_sublist.sort()
            for idx_image in range(5):
                image=image_sublist[idx_image]
                img_name = os.path.split(image)[-1]

                atten_map = Image.open(self.path_to_atten + img_name[:-4] + '.jpg')
                atten_map = atten_map.convert('L')
                atten_map = atten_map.resize((112, 112))
                atten_map =np.reshape(atten_map,[112,112,1])
                atten_map = np.asarray(atten_map, np.float64)  # [x,x,x]x=0-255
                atten_map = atten_map / 255.0

                polar = Image.open(self.path_to_polar  + img_name[:-4] + '.jpg')
                polar = polar.resize((224, 224))
                polar = np.asarray(polar, np.uint8)
                polar = polar / 255.0

                image = Image.open(image)
                image = image.resize((224, 224))
                image = np.asarray(image, np.uint8)
                image = image / 255.0

                Atten_map.append(atten_map)
                Polar_map.append(polar)
                img.append(image)
                if idx_image ==0:
                    year.append(int(0))
                else:
                    year.append(int(img_name[7:11])-int(os.path.split(image_sublist[idx_image-1])[-1][7:11]))
                count = count + 1
            assert count == 5
            img_batch.append(np.array(img))
            year_batch.append(np.array(year))
            Atten_map_batch.append(np.array(Atten_map))
            Polar_map_batch.append(np.array(Polar_map))
            label_batch.append(np.array(label))
            self.cursor += 1
        self.batch_order += 1
        return np.array(img_batch),np.array(year_batch),np.array(Atten_map_batch),np.array(Polar_map_batch),np.array(label_batch)


    def get_batch_1(self):  # Returns
        if self.cursor + self.batch_size > self.size:
            self.cursor = 0
            self.batch_order = 0
            np.random.shuffle(self.list_img_path)

        label_batch=[]
        for idx in range(self.batch_size):

            label=[]


            image_subpath=self.list_img_path[self.batch_order * 4 + idx]
            with open(self.path_to_label+image_subpath+'.txt', 'r') as f:
                K = f.readlines()
                for i_line in range(5):
                    line= K[i_line+1]
                    line = line.strip('\n')
                    line = int(line)
                    label.append(line)




            label_batch.append(np.array(label))
            self.cursor += 1
        self.batch_order += 1
        return np.array(label_batch)