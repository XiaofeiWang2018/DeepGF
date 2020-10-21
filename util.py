# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import math


e = math.e
thre=0.5

def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)


def _parse_function_vad(example_proto):
    keys_to_features = {'shape': tf.FixedLenFeature([], tf.string),
                        'image': tf.FixedLenFeature([], tf.string),
                        'GTlabel': tf.FixedLenFeature([], tf.int64),

                        }
    input_size = [224, 224]
    parsed_features = tf.parse_single_example(example_proto, keys_to_features, name='features')
    image = tf.decode_raw(parsed_features['image'], tf.uint8)
    shape = tf.decode_raw(parsed_features['shape'], tf.int32)
    GTlabel = parsed_features['GTlabel']
    image = tf.reshape(image, [shape[0], shape[1], 3])      # before sess.run, it is a tensor, with dtype of tf.uint8, from 0 to 255
    image = tf.image.resize_images(image,[input_size[0], input_size[1]])    #float 32  0-255
    image = image / 255.0
    #
    return image, GTlabel, shape

def top_k_accuracy(self, labels, k=1):
    batch_size = labels.get_shape().as_list()[0]
    right_1 = tf.to_float(tf.nn.in_top_k(predictions=self.predict_label, targets=labels, k=k))          #find the max value of prediction
    num_correct_per_batch = tf.reduce_sum(right_1)
    acc = num_correct_per_batch / batch_size
    tf.summary.scalar('acc', acc)
    return acc



class AlexNet (object):



    def __init__(self,input,num_classes,is_training = True,dropout_keep_prob = 0.5,spatial_squeeze = True):
        self.inputs=input
        self.num_classes=num_classes
        self.is_training = is_training
        self.dropout_keep_prob = dropout_keep_prob
        self.spatial_squeeze = spatial_squeeze  # 参数标志是否对输出进行squeeze操作（去除维度数为1的维度，比如5*3*1转为5*3）
        self.scope = 'alexnet'
        self.trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

    def alexnet_v2_arg_scope(weight_decay=0.0005,reuse=None):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,  # relu是AlexNet的一大亮点，取代了之前的softmax。
                            biases_initializer=tf.constant_initializer(0.1),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME'):  # 一般卷积的padding模式为SAME
                with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:  # pool的padding模式为VALID
                    return arg_sc



    def alexnet_5x5(self,inputs):

        """ fc层降维"""
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
            net1 = slim.conv2d(inputs, 32, [11, 11], 4, padding='VALID',scope='conv1') #54
            net2 = slim.max_pool2d(net1, [3, 3], 2, scope='pool1') #26
            net3 = slim.conv2d(net2, 64, [5, 5], scope='conv2')  #26
            net4 = slim.max_pool2d(net3, [3, 3], 2, scope='pool2') #12
            net5 = slim.conv2d(net4, 128, [3, 3], scope='conv3') #12
            net6 =net5
            net7 = net5
            net8 = slim.max_pool2d(net5, [3, 3], 2, scope='pool5') #5
            # Use conv2d instead of fully_connected layers.

            with slim.arg_scope([slim.fully_connected],
                                weights_initializer=self.trunc_normal(0.005),
                                biases_initializer=tf.constant_initializer(0.1)):
                conv_shape = net8.get_shape().as_list()
                net9 = tf.reduce_mean(net8,axis=(1,2))

                net10 = slim.fully_connected(net9, 512, scope='fc6')
                net11 = slim.dropout(net10, self.dropout_keep_prob, is_training=self.is_training,scope='dropout6')
                net12 = slim.fully_connected(net11, self.num_classes,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  biases_initializer=tf.zeros_initializer(),
                                  scope='fc7')

            return [net1,net2,net3,net4,net5,net6,net7,net8,net9,net10,net11,net12]

    def alexnet_12x12(self):

        """ fc层降维"""
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
            net1 = slim.conv2d(self.inputs, 32, [11, 11], 4, padding='VALID',scope='conv1') #54
            net2 = slim.max_pool2d(net1, [3, 3], 2, scope='pool1') #26
            net3 = slim.conv2d(net2, 64, [5, 5], scope='conv2')  #26
            net4 = slim.max_pool2d(net3, [3, 3], 2, scope='pool2') #12
            net5 = slim.conv2d(net4, 128, [3, 3], scope='conv3') #12
            net6 = net5
            net7 = net5
            net8 = net5
            # Use conv2d instead of fully_connected layers.

            with slim.arg_scope([slim.fully_connected],
                                weights_initializer=self.trunc_normal(0.005),
                                biases_initializer=tf.constant_initializer(0.1)):
                conv_shape = net8.get_shape().as_list()
                net9 = tf.reduce_mean(net8,axis=(1,2))

                net10 = slim.fully_connected(net9, 512, scope='fc6')
                net11 = slim.dropout(net10, self.dropout_keep_prob, is_training=self.is_training,scope='dropout6')
                net12 = slim.fully_connected(net11, self.num_classes,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  biases_initializer=tf.zeros_initializer(),
                                  scope='fc7')

            return [net1,net2,net3,net4,net5,net6,net7,net8,net9,net10,net11,net12]


    def alexnet_5x5_complex(self,inputs):


        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
            net1 = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',scope='conv1') #54
            net2 = slim.max_pool2d(net1, [3, 3], 2, scope='pool1') #26
            net3 = slim.conv2d(net2, 192, [5, 5], scope='conv2')  #26
            net4 = slim.max_pool2d(net3, [3, 3], 2, scope='pool2') #12
            net5 = slim.conv2d(net4, 384, [3, 3], scope='conv3') #12
            net6 = slim.conv2d(net5, 384, [3, 3], scope='conv4') #12
            net7 = slim.conv2d(net6, 256, [3, 3], scope='conv5') #12
            net8 = slim.max_pool2d(net7, [3, 3], 2, scope='pool5') #5
            # Use conv2d instead of fully_connected layers.


            return [net1,net2,net3,net4,net5,net6,net7,net8]


