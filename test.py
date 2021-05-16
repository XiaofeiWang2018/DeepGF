import tensorflow as tf
import random
import CNN_ALEX as Network
import time
import numpy as np
import math
import os
from matplotlib import pyplot as plt
from data_processing import DataLoader_atten_polar

mode='Ours'
test_num = 348
init_a=100
init_b=0
batch_size= 4
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)

def main():

    n_neurons = [5, 5]
    n_steps = 5
    n_layers = 2
    n_outputs = 2
    randon_seed = 510
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    random.seed(a=randon_seed)
    tf.set_random_seed(randon_seed)
    input_size = [224, 224]

    batch_size_val = batch_size

    net = Network.Net(
        batch_size=batch_size,
        n_steps=n_steps,
        n_layers=n_layers,
        n_neurons=n_neurons,
        n_outputs=n_outputs,
        init_lr=2*1e-4
    )
    net.trainable = True
    input = tf.placeholder(tf.float32, (batch_size, n_steps, 224, 224, 3))
    GT_label = tf.placeholder(tf.int64, (batch_size, n_steps))   #size = [batch, n_steps
    label_attention_map = tf.placeholder(tf.float32, (batch_size, n_steps, 112, 112, 1))
    label_polar_map = tf.placeholder(tf.float32, (batch_size, n_steps, 224, 224, 3))
    delta_year=tf.placeholder(tf.float32, (batch_size, n_steps))
    label_predict_op,a_1,b_1,a_2,b_2 = net.inference_model_8(input,label_attention_map,label_polar_map,delta_year,init_a,init_b)  # label_predict_op=(batch, n_steps, n_outputs)


    loss_per_batch=net._loss_per_batch(label_predict_op, GT_label)
    loss_op, loss_label_op, loss_weight_op = net._loss_liliu(label_predict_op, GT_label)   #[batch,n_steps]
    tf.summary.scalar('loss_op_'+mode, loss_op)
    acc_op = net._top_k_accuracy(label_predict_op, GT_label)





    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        list_img_path_test = os.listdir('./data/test/image/all')
        list_img_path_test.sort()
        """test"""
        dataloader_test = DataLoader_atten_polar(batch_size=batch_size,list_img_path=list_img_path_test, state='test')
        saver = tf.train.Saver()
        model_file = tf.train.latest_checkpoint('./pretrained_model/')
        saver.restore(sess, model_file)
        count = 0

        tp = 0.0
        fn = 0.0
        tn = 0.0
        fp = 0.0
        for j in range(int(test_num / batch_size_val)):
            imagev, yearv, GTmapv, Polarv, GTlabelv = dataloader_test.get_batch()
            label_predict = sess.run(label_predict_op, feed_dict={input: imagev, GT_label: GTlabelv,
                                                                  label_attention_map: GTmapv,
                                                                  label_polar_map: Polarv,
                                                                  delta_year: yearv
                                                                 })
            GTlabelv_test = np.reshape(GTlabelv, [-1])  # batch_size* n_steps
            label_predict_test = np.reshape(label_predict, [-1, 2])  # batch_size*n_steps,2
            label_predict_0 = label_predict_test[:, 0]  # batch_size, n_steps
            label_predict_1 = label_predict_test[:, 1]  # batch_size, n_steps

            """----------------------------tptn---------------------------------"""
            for nb in range(batch_size_val * n_steps):
                if GTlabelv_test[nb] == 1 and (label_predict_1[nb] > label_predict_0[nb]):
                    tp = tp + 1
                if GTlabelv_test[nb] == 0 and (label_predict_1[nb] < label_predict_0[nb]):
                    tn = tn + 1
                if GTlabelv_test[nb] == 1 and (label_predict_1[nb] < label_predict_0[nb]):
                    fn = fn + 1
                if GTlabelv_test[nb] == 0 and (label_predict_1[nb] > label_predict_0[nb]):
                    fp = fp + 1
            """----------------------------print all result of 384---------------------------------"""


        acc = (tp + tn) / (tp + tn + fp + fn)
        Sen = tp / (tp + fn)
        Spe = tn / (tn + fp)

        print("test accuracy: %.03f%% |test sen: %.03f%% |test spe: %.03f%%" % (
            100. * acc, 100. * Sen, 100. * Spe))




if __name__ == '__main__':


    main()


