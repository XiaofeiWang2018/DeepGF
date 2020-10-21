import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os
def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)


def _parse_function_vad(example_proto):
    keys_to_features = {'shape': tf.FixedLenFeature([], tf.string),
                        'image': tf.FixedLenFeature([], tf.string),
                        'GTlabel': tf.FixedLenFeature([], tf.int64)
                        }
    input_size = [224, 224]
    parsed_features = tf.parse_single_example(example_proto, keys_to_features, name='features')
    image = tf.decode_raw(parsed_features['image'], tf.uint8)
    shape = tf.decode_raw(parsed_features['shape'], tf.int32)       #original shape, example [500, 500]
    GTlabel = parsed_features['GTlabel']
    image = tf.reshape(image, [shape[0], shape[1], 3])      # before sess.run, it is a tensor, with dtype of tf.uint8, from 0 to 255
    image = tf.image.resize_images(image,[input_size[0], input_size[1]])    #float 32  0-255
    image = image / 255.0
    return image, GTlabel, shape,
def _parse_function_atten(example_proto):
    keys_to_features = {'GTmap':tf.FixedLenFeature([], tf.string),
                        'shape': tf.FixedLenFeature([], tf.string),
                        'image': tf.FixedLenFeature([], tf.string),
                        'name': tf.FixedLenFeature([], tf.string),
                        'GTlabel': tf.FixedLenFeature([], tf.int64)
                        }
    output_size = [112, 112]
    input_size = [224, 224]
    parsed_features = tf.parse_single_example(example_proto, keys_to_features, name='features')
    image = tf.decode_raw(parsed_features['image'], tf.uint8)
    shape = tf.decode_raw(parsed_features['shape'], tf.int32)       #original shape, example [500, 500]
    GTmap = tf.decode_raw(parsed_features['GTmap'], tf.float64)
    GTlabel = parsed_features['GTlabel']
    filename = tf.decode_raw(parsed_features['name'], tf.float32)

    image = tf.reshape(image, [shape[0], shape[1], 3])      # before sess.run, it is a tensor, with dtype of tf.uint8, from 0 to 255
    image = tf.image.resize_images(image,[input_size[0], input_size[1]])    #float 32  0-255
    image = image / 255.0
    GTmap = tf.reshape(GTmap, [112, 112, 1])
    GTmap = tf.image.resize_images(GTmap, [output_size[0], output_size[1]])     #float 32   0-1
    return image, GTmap, GTlabel, shape, filename

def top_k_accuracy(self, labels, k=1):
    batch_size = labels.get_shape().as_list()[0]
    right_1 = tf.to_float(tf.nn.in_top_k(predictions=self.predict_label, targets=labels, k=k))          #find the max value of prediction
    num_correct_per_batch = tf.reduce_sum(right_1)
    acc = num_correct_per_batch / batch_size
    tf.summary.scalar('acc', acc)
    return acc

class ChenNet(object):

    def __init__(self,input,num_classes,is_training = True,dropout_keep_prob = 0.5,spatial_squeeze = True):
        self.inputs=input
        self.num_classes=num_classes
        self.is_training = is_training
        self.dropout_keep_prob = dropout_keep_prob
        self.spatial_squeeze = spatial_squeeze  # 参数标志是否对输出进行squeeze操作（去除维度数为1的维度，比如5*3*1转为5*3）
        self.scope = 'chennet'
        self.trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

    def chennet_v2_arg_scope(weight_decay=0.0005):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,  # relu是AlexNet的一大亮点，取代了之前的softmax。
                            biases_initializer=tf.constant_initializer(0.1),
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.conv2d], padding='SAME'):  # 一般卷积的padding模式为SAME
                with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:  # pool的padding模式为VALID
                    return arg_sc


    def chennet_v2(self):


        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
            net1 = slim.conv2d(self.inputs, 64, [11, 11], 4, padding='VALID',scope='conv1') #55
            net2 = slim.max_pool2d(net1, [3, 3], 2, scope='pool1') #27
            net3 = slim.conv2d(net2, 192, [5, 5], scope='conv2')  #27
            net4 = slim.max_pool2d(net3, [3, 3], 2, scope='pool2') #13
            net5 = slim.conv2d(net4, 192, [3, 3], scope='conv3') #13
            net6 = slim.max_pool2d(net5, [3, 3], 2, scope='pool5') #6
            # Use conv2d instead of fully_connected layers.
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=self.trunc_normal(0.005),
                                biases_initializer=tf.constant_initializer(0.1)):
                net7 = slim.conv2d(net6, 4096, [5, 5], padding='VALID',
                                  scope='fc6')
                net8 = slim.dropout(net7, self.dropout_keep_prob, is_training=self.is_training,
                                   scope='dropout6')
                net9 = slim.conv2d(net8, 4096, [1, 1], padding='VALID', scope='fc7')
                net10 = slim.dropout(net9, self.dropout_keep_prob, is_training=self.is_training,
                                   scope='dropout7')
                net11 = slim.conv2d(net10, self.num_classes, [1, 1],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  biases_initializer=tf.zeros_initializer(),
                                  scope='fc8')

            # Convert end_points_collection into a end_point dict.

            if self.spatial_squeeze:
                net12 = tf.squeeze(net11, [1, 2], name='fc8/squeezed')  # 见后文详细注释

            return [net1,net2,net3,net4,net5,net6,net7,net8,net9,net10,net11,net12]