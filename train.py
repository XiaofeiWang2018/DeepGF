import tensorflow as tf
import random
import CNN_ALEX as Network
import time
import numpy as np
import math
import os
from matplotlib import pyplot as plt
from data_processing import DataLoader_atten_polar
from sklearn.metrics import roc_auc_score
import platform

mode='Performance drops v.s less parameters -- no lstm'
test_num = 348
cnn_pretrained_path='./CVPR_256_polar_64/'
lr_change= [1*1e-7, 4*1e-7, 4*1e-6, 4*1e-6, 4*1e-5, 2*1e-4]
throw_rate=0.5
trainingset_num=[1308,652,324,160,84,84]
num_strategy=5
strategy_epoch_duration=[3,3,3,3,3,3]
Epoch = 300
epoch_test = 2
save_epoch =20
init_a=100
init_b=0
batch_size= 4
randon_seed = 711
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
summary_save_dir= "summary51/"
acc_txt="acc51.txt"
log_txt="log51.txt"
test_details="test_detail51.txt"
config_txt="config51.txt"
throw_txt="throw51.txt"
save_path='model51'
with open("config51.txt", "w+")as f0:
    f0.write('mode: ' + mode)
    f0.write('\n')
    f0.flush()
    f0.write('throw_rate: %.3f ' % throw_rate)
    f0.write('\n')
    f0.flush()
    f0.write('num_strategy: %d ' % num_strategy)
    f0.write('\n')
    f0.flush()
    f0.write('epoch_test: %d' % epoch_test)
    f0.write('\n')
    f0.flush()
    f0.write('init_a: %.3f ' % init_a)
    f0.write('\n')
    f0.flush()
    f0.write('All epoch: %d ' % Epoch)
    f0.write('\n')
    f0.flush()


def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)

def main():
    g1 = tf.Graph()
    with g1.as_default():
        n_neurons = [5, 5]
        n_steps = 5
        n_layers = 2
        n_outputs = 2

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf.set_random_seed(randon_seed)
        input_size = [224, 224]

        batch_size_val = batch_size

        net = Network.Net(
            batch_size=batch_size,
            n_steps=n_steps,
            n_layers=n_layers,
            n_neurons=n_neurons,
            n_outputs=n_outputs,
            init_lr=lr_change[0]
        )
        net.trainable = True
        input = tf.placeholder(tf.float32, (batch_size, n_steps, 224, 224, 3))
        GT_label = tf.placeholder(tf.int64, (batch_size, n_steps))   #size = [batch, n_steps
        label_attention_map = tf.placeholder(tf.float32, (batch_size, n_steps, 112, 112, 1))
        label_polar_map = tf.placeholder(tf.float32, (batch_size, n_steps, 224, 224, 3))
        delta_year=tf.placeholder(tf.float32, (batch_size, n_steps))
        label_predict_op = net.inference_model_1(input,label_attention_map,label_polar_map,delta_year,init_a,init_b)  # label_predict_op=(batch, n_steps, n_outputs)
        lr = tf.placeholder(tf.float32, shape=[])

        loss_per_batch=net._loss_per_batch(label_predict_op, GT_label)
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
        saver = tf.train.Saver(max_to_keep=50)

        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(summary_save_dir+"train", sess.graph)
            test_writer = tf.summary.FileWriter(summary_save_dir+"test", sess.graph)

            list_img_path_train_init = os.listdir('./data/train/image/all')
            list_img_path_train_init.sort()
            list_img_path_train=list_img_path_train_init
            list_img_path_test = os.listdir('./data/test/image/all')
            list_img_path_test.sort()
            """train"""
            dataloader_train = DataLoader_atten_polar(batch_size=batch_size, list_img_path= list_img_path_train,state='train')
            """tensorboard"""
            dataloader_tensorboard = DataLoader_atten_polar(batch_size=batch_size, list_img_path= list_img_path_test, state='test')
            """test"""
            dataloader_test = DataLoader_atten_polar(batch_size=batch_size,list_img_path=list_img_path_test, state='test')
            """train_strategy"""
            dataloader_strategy = DataLoader_atten_polar(batch_size=batch_size, list_img_path=list_img_path_train, state='train')

            # print(sess.run('label_attention/1conv_1/weights:0'))
            # variables1 = tf.contrib.framework.get_variables_to_restore()[0:216]
            # saver_pretrainedCNN = tf.train.Saver(variables1)
            # model_file1 = tf.train.latest_checkpoint(cnn_pretrained_path)
            # saver_pretrainedCNN.restore(sess, model_file1)
            # print(sess.run('polar/1block_4_1/E2/weights/Adam:0'))
            count = 0
            count_strategy=0
            flag_strategy = True
            print("Start Training, model1!")
            with open(acc_txt, "w+") as f:
                with open(log_txt, "w+")as f2:
                    for epoch in range(0, Epoch): #3*10
                        print('\nEpoch: %d' % (epoch + 1))
                        """-----------------------------------------throw------------------------------------------"""
                        if epoch>0 and  (epoch % strategy_epoch_duration[count_strategy]) == 0 and flag_strategy:
                            if count_strategy == (num_strategy-1):
                                flag_strategy = False

                            count_strategy += 1  # 1~10
                            train_leftnum_before = trainingset_num[count_strategy-1]
                            train_leftnum_now = trainingset_num[count_strategy]
                            loss_trainingset = [0 for i_loss_train in range(train_leftnum_before)]
                            list_img_path_train_before = list_img_path_train

                            for jj in range(int(train_leftnum_before / batch_size)):
                                image_train_test, year_train_test, GTmap_train_test, Polar_train_test, GTlabel_train_test = dataloader_strategy.get_batch()
                                # loss_per_batch_1:[batch_size]
                                loss_per_batch_1 = sess.run(loss_per_batch, feed_dict={input: image_train_test,
                                                                                       GT_label: GTlabel_train_test,
                                                                                       label_attention_map: GTmap_train_test,
                                                                                       label_polar_map: Polar_train_test,
                                                                                       delta_year: year_train_test,
                                                                                       lr:lr_change[count_strategy]})
                                # loss_trainingset:[train_leftnum]
                                for i_loss_chosen in range(batch_size):
                                    loss_trainingset[jj * 4 + i_loss_chosen] = loss_per_batch_1[i_loss_chosen]

                            matrix1 = np.zeros([train_leftnum_before, 2], dtype=np.float)
                            matrix2 = np.zeros([train_leftnum_now, 2], dtype=np.float)
                            for i_1 in range(train_leftnum_before):
                                matrix1[i_1] = [loss_trainingset[i_1], i_1]
                            matrix1 = sorted(matrix1, key=lambda cus: cus[0], reverse=True)  # big-->small

                            for i_2 in range(train_leftnum_now):
                                matrix2[i_2] = matrix1[i_2]
                            for i_3 in range(train_leftnum_now):
                                if i_3 == 0:
                                    list_img_path_train0 = [list_img_path_train_before[int(matrix2[i_3][1])]]
                                else:
                                    list_img_path_train0.append(list_img_path_train_before[int(matrix2[i_3][1])])
                            list_img_path_train = list_img_path_train0

                            with open(throw_txt, "a")as f3:
                                f3.write('strategy: %d ' % count_strategy)
                                f3.write('\n')
                                f3.flush()
                                for i_f3 in range(train_leftnum_before-train_leftnum_now):
                                    f3.write(list_img_path_train_before[int(matrix1[train_leftnum_now+i_f3][1])]+' : ')
                                    f3.flush()
                                    with open('data/train/label/all/'+ list_img_path_train_before[int(matrix1[train_leftnum_now+i_f3][1])] + '.txt', 'r') as f4:
                                        K = f4.readlines()
                                        for i_line in range(5):
                                            line = K[i_line + 1]
                                            line = line.strip('\n')
                                            line = int(line)
                                            f3.write(str(line))
                                            f3.flush()

                                    f3.write('\n')
                                    f3.flush()
                                if count_strategy == num_strategy:
                                    f3.write('Last left: ')
                                    f3.write('\n')
                                    f3.flush()
                                    for i_f3_lastleft in range(train_leftnum_now):
                                        f3.write(list_img_path_train[i_f3_lastleft] + ' : ')
                                        f3.flush()
                                        with open('data/train/label/all/' + list_img_path_train[i_f3_lastleft] + '.txt', 'r') as f5:
                                            K = f5.readlines()
                                            for i_line in range(5):
                                                line = K[i_line + 1]
                                                line = line.strip('\n')
                                                line = int(line)
                                                f3.write(str(line))
                                                f3.flush()
                                        f3.write('\n')
                                        f3.flush()


                            dataloader_train = DataLoader_atten_polar(batch_size=batch_size,list_img_path=list_img_path_train, state='train')
                            dataloader_strategy = DataLoader_atten_polar(batch_size=batch_size,list_img_path=list_img_path_train, state='train')
                        """-----------------------------------------train------------------------------------------"""
                        for i in range(int(trainingset_num[count_strategy]/batch_size)):
                            image1,year1,GTmap1,Polar1, GTlabel1= dataloader_train.get_batch()
                            loss_train,_, acc,label_predict = sess.run([ loss_op,train_op, acc_op,label_predict_op],feed_dict=
                            {input: image1, GT_label: GTlabel1,label_attention_map:GTmap1,label_polar_map:Polar1,delta_year:year1,lr:lr_change[count_strategy]})

                            print('cls [strategy:%d, epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' % (
                                count_strategy, epoch,(i + 1), loss_train, 100. * acc))

                            f2.write('strategy:%d, epoch:%03d  %05d |Loss: %.03f | Acc: %.3f%%| a1: %.3f| b1: %.3f| a2: %.3f| b2: %.3f ' % (
                                count_strategy ,epoch , (i + 1), loss_train, 100. * acc,0,0,0,0))
                            f2.write('\n')
                            f2.flush()
                            count +=1

                            if count % 20 == 0:  # tensorboard
                                image2, year2, GTmap2, Polar2, GTlabel2=dataloader_tensorboard.get_batch()
                                train_s = sess.run(summary_op,feed_dict={input: image1, GT_label: GTlabel1,label_attention_map:GTmap1,label_polar_map:Polar1,delta_year:year1,lr:lr_change[count_strategy]})
                                train_writer.add_summary(train_s, count)
                                test_s = sess.run(summary_op,feed_dict={input: image2, GT_label: GTlabel2,label_attention_map:GTmap2,label_polar_map:Polar2,delta_year:year2,lr:lr_change[count_strategy]})
                                test_writer.add_summary(test_s, count)
                        """-----------------------------------------test------------------------------------------"""
                        with open(test_details, "a")as f6:
                            f6.write('epoch: %d' % epoch)
                            f6.write('\n')
                            f6.flush()
                            if epoch % epoch_test == 0:
                                print("testing")
                                tp = 0.0
                                fn = 0.0
                                tn = 0.0
                                fp = 0.0
                                y_true = [0.0 for _ in range(test_num * n_steps)]
                                y_scores = [0.0 for _ in range(test_num * n_steps)]
                                for j in range(int(test_num / batch_size_val)):
                                    imagev, yearv, GTmapv, Polarv, GTlabelv = dataloader_test.get_batch()
                                    label_predict = sess.run(label_predict_op,
                                                             feed_dict={input: imagev, GT_label: GTlabelv,
                                                                        label_attention_map: GTmapv,
                                                                        label_polar_map: Polarv,
                                                                        delta_year: yearv
                                                                        })
                                    GTlabelv_test = np.reshape(GTlabelv, [-1])  # batch_size* n_steps
                                    label_predict_test = np.reshape(label_predict, [-1, 2])  # batch_size*n_steps,2
                                    label_predict_0 = label_predict_test[:, 0]  # batch_size* n_steps
                                    label_predict_1 = label_predict_test[:, 1]  # batch_size* n_steps

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
                                    """----------------------------AUC---------------------------------"""
                                    for nb in range(batch_size_val * n_steps):  # 20
                                        y_true[j * (batch_size_val * n_steps) + nb] = GTlabelv_test[nb]
                                        y_scores[j * (batch_size_val * n_steps) + nb] = (math.exp(
                                            label_predict_1[nb])) / (math.exp(label_predict_1[nb]) + math.exp(
                                            label_predict_0[nb]))
                                    """----------------------------print all result of 384---------------------------------"""
                                    for batch in range(batch_size_val):
                                        f6.write('%s   :' % list_img_path_test[j * 4 + batch])
                                        f6.flush()
                                        for img_perimgpath in range(5):
                                            f6.write('%d' % GTlabelv[batch][img_perimgpath])
                                            f6.flush()
                                        f6.write(' ')
                                        f6.flush()
                                        for img_perimgpath in range(5):
                                            if label_predict[batch][img_perimgpath][1] > \
                                                    label_predict[batch][img_perimgpath][0]:
                                                f6.write('1')
                                                f6.flush()
                                            if label_predict[batch][img_perimgpath][1] < \
                                                    label_predict[batch][img_perimgpath][0]:
                                                f6.write('0')
                                                f6.flush()
                                        f6.write('   ')
                                        f6.flush()
                                        for img_perimgpath in range(5):
                                            p_glau = math.exp(label_predict[batch][img_perimgpath][1]) / (
                                                        math.exp(label_predict[batch][img_perimgpath][1]) + math.exp(
                                                    label_predict[batch][img_perimgpath][0]))
                                            f6.write('%.03f%% ' % (100. * p_glau))
                                            f6.flush()
                                        f6.write('\n')
                                        f6.flush()

                                acc = (tp + tn) / (tp + tn + fp + fn)
                                Sen = tp / (tp + fn)
                                Spe = tn / (tn + fp)

                                AUC = roc_auc_score(y_true, y_scores)
                                print("test accuracy: %.03f%% |test sen: %.03f%% |test spe: %.03f%%" % (
                                    100. * acc, 100. * Sen, 100. * Spe))
                                f.write(
                                    ' epoch %03d  | Acc: %.3f%% | sen: %.3f%% | spe: %.3f%% |AUC: %.3f%%|tp: %.3f | tn: %.3f| fp: %.3f | fn: %.3f' % (
                                        epoch, 100. * acc, 100. * Sen, 100. * Spe, 100. * AUC, tp, tn,
                                        fp, fn))
                                f.write('\n')
                                f.flush()
                                dataloader_test = DataLoader_atten_polar(batch_size=batch_size,
                                                                         list_img_path=list_img_path_test, state='test')
                        if  epoch >0 and (epoch %save_epoch)== 0:  # ckpt
                            saver.save(sess, save_path+'/save.ckpt', global_step=epoch )


if __name__ == '__main__':
    remove_all_file(summary_save_dir+'test')
    remove_all_file(summary_save_dir+'train')
    remove_all_file(save_path)
    if platform.system() =='Linux':
        if (os.path.exists(throw_txt)):
            os.remove(throw_txt)
        os.mknod(throw_txt)
        if (os.path.exists(test_details)):
            os.remove(test_details)
        os.mknod(test_details)
    main()


