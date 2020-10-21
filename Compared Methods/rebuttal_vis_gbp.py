"""rebuttal visualization """
import argparse
import tensorflow as tf
import random
import CNN_ALEX as Network
import time
import numpy as np
import math
import os
from matplotlib import pyplot as plt
from data_processing import DataLoader_atten_polar

from saliency.grad_cam import *
import saliency
from matplotlib import image as Image

parser = argparse.ArgumentParser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser.add_argument('--task', type=str, default='cls')

model_file_cls = 'models11/save.ckpt-321'

map_01 = [3, 2, 2, 4, 3]
map_11 = [4, 3, 4, 5, 4]

H = 224


def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)


def gate(input, gate=0.5):
    for i in range(H):
        for j in range(H):
            if input[i, j] < gate:
                input[i, j] = 0
    return input


def main(args):
    """------------------ parse input--------------------"""

    task = args.task

    # remove_all_file('visualization_result/' + task+'/'+vis_mode)

    """---------------------------------------------------"""
    with tf.Graph().as_default() as graph:
        lr = 2 * 1e-4
        batch_size = 1
        n_neurons = [5, 5]
        n_steps = 5
        n_layers = 2
        n_outputs = 2
        init_a = 100
        init_b = 0
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        net = Network.Net(
            batch_size=1,
            n_steps=n_steps,
            n_layers=n_layers,
            n_neurons=n_neurons,
            n_outputs=n_outputs,
            init_lr=lr
        )
        net.trainable = True
        input = tf.placeholder(tf.float32, (batch_size, n_steps, 224, 224, 3))
        GT_label = tf.placeholder(tf.int64, (batch_size, n_steps))  # size = [batch, n_steps
        label_attention_map = tf.placeholder(tf.float32, (batch_size, n_steps, 112, 112, 1))
        label_polar_map = tf.placeholder(tf.float32, (batch_size, n_steps, 224, 224, 3))
        delta_year = tf.placeholder(tf.float32, (batch_size, n_steps))
        label_predict_op, cnn_out = net.inference_model_10(input,
                                                           label_attention_map)  # label_predict_op=(batch, n_steps, n_outputs)
        lr = tf.placeholder(tf.float32, shape=[])

        loss_per_batch = net._loss_per_batch(label_predict_op, GT_label)
        loss_op, loss_label_op, loss_weight_op = net._loss_liliu(label_predict_op, GT_label)  # [batch,n_steps]

        acc_op = net._top_k_accuracy(label_predict_op, GT_label)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            opt = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
            gradients = opt.compute_gradients(loss_op)  # all variables are trainable
            apply_gradient_op = opt.apply_gradients(gradients)  # , global_step=self.global_step
            train_op = apply_gradient_op

        with tf.Session(config=tf_config, graph=graph) as sess:
            sess.run(tf.global_variables_initializer())

            list_img_path_test = os.listdir('./data/test_vis/image/all')
            list_img_path_test.sort()
            dataloader_test = DataLoader_atten_polar(batch_size=batch_size, list_img_path=list_img_path_test,
                                                     state='test')

            for i in range(5):


                """----------------------01------------------------------------"""
                # image, year, GTmap, Polar, GTlabel = dataloader_test.get_batch_single()
                # saver = tf.train.Saver()
                # saver.restore(sess, model_file_cls)
                #
                # guided_backprop = saliency.GradientSaliency(graph, sess, label_predict_op[0, :, 1], input)
                # mask0 = guided_backprop.GetMask(input[0], feed_dict={input: image, GT_label: GTlabel,
                #                                                      label_attention_map: GTmap,
                #                                                      label_polar_map: Polar,
                #                                                      delta_year: year})
                # jump = map_01[i]
                #
                # mask = mask0[jump - 2][0][jump - 2]
                # mask = saliency.VisualizeImageGrayscale(mask)
                # mask = gate(mask, gate=0)
                # Image.imsave('visualization_result/cls_cvpr/final/01/img' + str(i + 1) + '_jump_cls.jpg', mask,
                #              cmap=plt.get_cmap('gray'))
                # Image.imsave(
                #     'visualization_result/cls_cvpr/final/01/img' + str(i + 1) + '.jpg',
                #     image[0][jump - 2])
                # print(str(i + 1))
                """----------------------11------------------------------------"""
                image, year, GTmap, Polar, GTlabel = dataloader_test.get_batch_single()
                saver = tf.train.Saver()
                saver.restore(sess, model_file_cls)

                guided_backprop = saliency.GradientSaliency(graph, sess, label_predict_op[0, :, 1], input)
                mask0 = guided_backprop.GetMask(input[0], feed_dict={input: image, GT_label: GTlabel,
                                                                     label_attention_map: GTmap,
                                                                     label_polar_map: Polar,
                                                                     delta_year: year})
                jump = map_11[i]

                mask = mask0[jump - 2][0][jump - 2]
                mask = saliency.VisualizeImageGrayscale(mask)
                mask = gate(mask, gate=0)
                Image.imsave('visualization_result/cls_cvpr/final/11/img' + str(i + 1) + '_glau_cls.jpg', mask,
                             cmap=plt.get_cmap('gray'))
                Image.imsave(
                    'visualization_result/cls_cvpr/final/11/img' + str(i + 1) + '.jpg',
                    image[0][jump - 2])
                print(str(i + 1))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
