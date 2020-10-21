import tensorflow as tf
import random
import CNN_ALEX as Network
import time
import numpy as np
import math
import os
from matplotlib import pyplot as plt
from data_processing import DataLoader_atten_polar

mode='strategy-->data augmentation'

lr=4*1e-7
Epoch = 50
epoch_test = 4
init_a=100
init_b=0
batch_size= 4
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
summary_save_dir= "summary2/"
acc_txt="acc2.txt"
log_txt="log2.txt"
cnn_pretrained_path='../N19_our_model/CVPR_256_polar_64/'



def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)

def main():
    sess = tf.InteractiveSession()
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
    test_num = 348
    net = Network.Net(
        batch_size=batch_size,
        n_steps=n_steps,
        n_layers=n_layers,
        n_neurons=n_neurons,
        n_outputs=n_outputs,
        init_lr=lr
    )
    net.trainable = True
    input = tf.placeholder(tf.float32, (batch_size, n_steps, 224, 224, 3))
    GT_label = tf.placeholder(tf.int64, (batch_size, n_steps))   #size = [batch, n_steps
    label_attention_map = tf.placeholder(tf.float32, (batch_size, n_steps, 112, 112, 1))
    label_polar_map = tf.placeholder(tf.float32, (batch_size, n_steps, 224, 224, 3))
    delta_year=tf.placeholder(tf.float32, (batch_size, n_steps))
    label_predict_op,a_1,b_1,a_2,b_2 = net.inference_model_8(input,label_attention_map,label_polar_map,delta_year,init_a,init_b)  # label_predict_op=(batch, n_steps, n_outputs)


    loss_op, loss_label_op, loss_weight_op = net._loss_liliu(label_predict_op, GT_label)   #[batch,n_steps]
    tf.summary.scalar('loss_op_'+mode, loss_op)
    acc_op = net._top_k_accuracy(label_predict_op, GT_label)
    tf.summary.scalar('accurancy', acc_op)

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        opt = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
        gradients = opt.compute_gradients(loss_op)  # all variables are trainable
        apply_gradient_op = opt.apply_gradients(gradients)  # , global_step=self.global_step
        train_op = apply_gradient_op
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=20)

    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(summary_save_dir+"train", sess.graph)
        test_writer = tf.summary.FileWriter(summary_save_dir+"test", sess.graph)

        list_img_path_train_init = os.listdir('../N19_our_model/data/train/image/all')
        list_img_path_train_init.sort()
        list_img_path_train_aug = []
        for i in range(len(list_img_path_train_init)):  # 1308
            label = 0
            list_img_path_train_aug.append(list_img_path_train_init[i])
            with open('../N19_our_model/data/train/label/all/' + list_img_path_train_init[i] + '.txt', 'r') as f:
                K = f.readlines()
                for i_line in range(5):
                    line = K[i_line + 1]
                    line = line.strip('\n')
                    line = int(line)
                    label += line
            if label >= 1 :
                for j in range(38):
                    list_img_path_train_aug.append(list_img_path_train_init[i])

        list_img_path_train_init = list_img_path_train_aug

        list_img_path_train=list_img_path_train_init
        list_img_path_test = os.listdir('../N19_our_model/data/test/image/all')
        list_img_path_test.sort()
        """train"""
        dataloader_train = DataLoader_atten_polar(batch_size=batch_size, list_img_path= list_img_path_train,state='train')
        """tensorboard"""
        dataloader_tensorboard = DataLoader_atten_polar(batch_size=batch_size, list_img_path= list_img_path_test, state='test')
        """test"""
        dataloader_test = DataLoader_atten_polar(batch_size=batch_size,list_img_path=list_img_path_test, state='test')

        # print(sess.run('conv1/weights:0'))
        variables1 = tf.contrib.framework.get_variables_to_restore()[0:216]
        saver_pretrainedCNN = tf.train.Saver(variables1)
        model_file1 = tf.train.latest_checkpoint(cnn_pretrained_path)
        saver_pretrainedCNN.restore(sess, model_file1)
        # print(sess.run('conv1/weights:0'))
        count = 0
        count_strategy=0

        for kk in range(int(len(list_img_path_train)/batch_size)):
            image1, year1, GTmap1, Polar1, GTlabel1 = dataloader_train.get_batch()

        print("Start Training, model1!")
        with open(acc_txt, "w+") as f:
            with open(log_txt, "w+")as f2:
                for epoch in range(0, Epoch): #3*10
                    print('\nEpoch: %d' % (epoch + 1))
                    """-----------------------------------------train------------------------------------------"""
                    for i in range(int(len(list_img_path_train)/batch_size)):
                        image1,year1,GTmap1,Polar1, GTlabel1= dataloader_train.get_batch()
                        loss_train,_, acc,label_predict,a1,b1,a2,b2 = sess.run([ loss_op,train_op, acc_op,label_predict_op,a_1,b_1,a_2,b_2],feed_dict=
                        {input: image1, GT_label: GTlabel1,label_attention_map:GTmap1,label_polar_map:Polar1,delta_year:year1})

                        print('dis'+acc_txt[3:-4]+ ' lr:'+str(lr)+' [strategy:%d, epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' % (
                            count_strategy, epoch,(i + 1), loss_train, 100. * acc))

                        f2.write(' lr:'+str(lr)+' strategy:%d, epoch:%03d  %05d |Loss: %.03f | Acc: %.3f%%| a1: %.3f| b1: %.3f| a2: %.3f| b2: %.3f ' % (
                            count_strategy ,epoch , (i + 1), loss_train, 100. * acc,a1,b1,a2,b2))
                        f2.write('\n')
                        f2.flush()
                        count +=1

                        if count % 20 == 0:  # tensorboard
                            image2, year2, GTmap2, Polar2, GTlabel2=dataloader_tensorboard.get_batch()
                            train_s = sess.run(summary_op,feed_dict={input: image1, GT_label: GTlabel1,label_attention_map:GTmap1,label_polar_map:Polar1,delta_year:year1})
                            train_writer.add_summary(train_s, count)
                            test_s = sess.run(summary_op,feed_dict={input: image2, GT_label: GTlabel2,label_attention_map:GTmap2,label_polar_map:Polar2,delta_year:year2})
                            test_writer.add_summary(test_s, count)
                    """-----------------------------------------test------------------------------------------"""

                    if epoch % epoch_test == 0:
                        print("testing")
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

                        acc = (tp + tn) / (tp + tn + fp + fn)
                        Sen = tp / (tp + fn)
                        Spe = tn / (tn + fp)
                        print("test accuracy: %.03f%% |test sen: %.03f%% |test spe: %.03f%% " % (
                            100. * acc, 100. * Sen, 100. * Spe))
                        f.write(
                            ' epoch %03d  | Acc: %.3f%% | sen: %.3f%% | spe: %.3f%% |  tp: %.3f | tn: %.3f| fp: %.3f | fn: %.3f' % (
                                epoch, 100. * acc, 100. * Sen, 100. * Spe, tp, tn, fp, fn))
                        f.write('\n')
                        f.flush()



if __name__ == '__main__':
    remove_all_file(summary_save_dir+'test')
    remove_all_file(summary_save_dir+'train')

    main()


