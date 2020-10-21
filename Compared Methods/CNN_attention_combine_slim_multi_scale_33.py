
import BasicNet_3
import tensorflow as tf
import math
from tensorflow.python.framework import ops

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    gate_g = tf.cast(grad > 0, "float32")
    #print('grad',sess.run(grad))
    gate_y = tf.cast(op.outputs[0] > 0, "float32")
    #print('relu_input',sess.run(op.outputs[0]))
    return grad * gate_g * gate_y#tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))


class Net(BasicNet_3.BasicNet):

    cell_size = 7
    salmask_lb_input = 0.2  # mask cam be salmask_lb~1
    salmask_lb_feature = 0.5
    eps = 1e-7



    init_learning_rate = 10 ** (-4)
    #global_step = tf.Variable(0, trainable=False)
    trainable = False
    batch_size_val = 8
    thre = 0.5
    Is_training = True
    wgn_r = 70

    def __init__(self):
        super(Net, self).__init__()

        #self.global_step = tf.Variable(0, trainable= False)
        #self.initial_var_collection.append(self.global_step)   #???
        self.predict_label = []
        self.predict_label_f =[]
        self.predict_salmap = []
        # self.acc = []
        self.mask_input =[]
        self.loss = []
        self.loss_salency = []
        self.loss_label = []
        self.loss_label_f =[]
        self.loss_weight = []
        self.loss_semi = []
        self.test =[]
        # self.loss_deta_semi = []
        # self.loss_label_semi = []
        # self.loss_weight_semi = []
        self.train = []
        self.train_semi = []
        self.fc2 = []
        self.beta1 = 10
        self.beta2 = 1
        self.beta3 = 1

    def inference(self, x, g, reuse= None):
        feature = self.get_feature_inference(x, reuse=reuse)
        salency_map = self.SalencyMap(feature, reuse=reuse)
        salency_map_norm = self._normlized_0to1(salency_map)
        self.predict_salmap = salency_map_norm

        # x1 = self.wgn(x,self.wgn_r)
        #         # feature_1 = self.get_feature_inference(x1, reuse=True)
        #         # salency_map_1 = self.SalencyMap(feature_1, reuse=True)
        #         # salency_map_norm_1 = self._normlized_0to1(salency_map_1)
        #         # self.predict_salmap_1 = salency_map_norm_1
        salency_mask_input = salency_map_norm * (1-self.salmask_lb_input) + self.salmask_lb_input
        salency_mask_feature = salency_map_norm * (1 - self.salmask_lb_feature) + self.salmask_lb_feature

        predict_label, mask_input, cnn_output = self.predict_inference(x, salency_mask_input, salency_mask_feature, reuse=reuse)    #[batch,num_labels],with out sigmoid to 0-1  , cnn_output_before, cnn_output_after

        self.predict_label = predict_label      #fc output
        label_attention_map = self._grad_bp(predict_label, x, g) #[batch,w,h,3]      #对输入求导还是对输入和mask求导？（batch,w,h,3）

        #对求导得到的特征图，先取绝对值，在从3通道平均为1通道，归一化，最后为[batch, w, h, 1]
        abs_map = tf.abs(label_attention_map)
        mat_shape = abs_map.get_shape().as_list()
        label_map = (abs_map[:,:,:,0] + abs_map[:,:,:,1] + abs_map[:,:,:,2])/3
        label_map = tf.reshape(label_map, [mat_shape[0], mat_shape[1], mat_shape[2], 1])
        gradient_255_max_pooling = tf.nn.max_pool(label_map, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1],
                                                  padding='SAME')
        gradient_255_ave_pooling = tf.nn.avg_pool(gradient_255_max_pooling, ksize=[1, 16, 16, 1], strides=[1, 1, 1, 1],
                                                  padding='SAME')

        #gradient_255_ave_pooling = tf.image.resize_images(gradient_255_ave_pooling, [112, 112])
        #_loss_double_mask
        label_attention_map_norm = self._normlized_0to1(gradient_255_ave_pooling)



        return salency_map_norm, predict_label, cnn_output, mask_input, label_attention_map_norm

        # predict_label_f#predict_label_f, cnn_output_before, cnn_output_after#salency_map_norm_1,
            #返回预测的salmap(0-1),fc的输出，卷积层的输出（用于cam），input image mask 上 pred salmap（-1，1）


    def get_label_attention_polar_inference(self, x, label_attention_map_norm,polar,reuse):
        label_attention_map_norm = label_attention_map_norm * (1 - 0.2) + 0.2
        with tf.variable_scope('label_attention'):
            atten_out = self.predict_inference(x, label_attention_map_norm, label_attention_map_norm,
                                               reuse=reuse)  # [:,:,:,0:1]
        with tf.variable_scope('polar'):
            polar_out = self.polar_inference(polar, reuse=reuse)  # [:,:,:,0:1]

        cn_out = tf.concat([atten_out, polar_out], axis=3)
        # cn_out=tf.reshape(cn_out,shape=[-1,cn_out.get_shape().as_list()[1]*cn_out.get_shape().as_list()[2]*cn_out.get_shape().as_list()[3]])
        cn_out=tf.reduce_mean(cn_out,axis=(1,2))
        stacked_outputs_0 = tf.layers.dense(cn_out, 256, name='fc0', reuse=reuse)
        stacked_outputs_0 = tf.layers.dropout(stacked_outputs_0, training=True, name='do0')
        fc1 = tf.layers.dense(stacked_outputs_0, 2, name='fc1', reuse=reuse)
        self.predict_label_f = fc1
        return cn_out,fc1

    def get_label_attention_inference_fc(self, x, label_attention_map_norm,reuse):
        label_attention_map_norm = label_attention_map_norm * (1 - 0.2) + 0.2
        with tf.variable_scope('label_attention'):
            atten_out = self.predict_inference(x, label_attention_map_norm, label_attention_map_norm,
                                               reuse=reuse)  # [:,:,:,0:1]
        # atten_out=tf.reshape(atten_out,shape=[-1,atten_out.get_shape().as_list()[1]*atten_out.get_shape().as_list()[2]*atten_out.get_shape().as_list()[3]])
        atten_out_ori=atten_out
        atten_out = tf.reduce_mean(atten_out, axis=(1, 2))
        stacked_outputs_0 = tf.layers.dense(atten_out, 256, name='fc0', reuse=reuse)
        stacked_outputs_0 = tf.layers.dropout(stacked_outputs_0, training=True, name='do0')
        fc1 = tf.layers.dense(stacked_outputs_0, 2, name='fc1', reuse=reuse)
        return fc1,atten_out_ori

    def get_label_attention_inference(self, x, label_attention_map_norm,reuse):
        label_attention_map_norm = label_attention_map_norm * (1 - 0.2) + 0.2
        with tf.variable_scope('label_attention'):
            atten_out = self.predict_inference(x, label_attention_map_norm, label_attention_map_norm,
                                               reuse=reuse)  # [:,:,:,0:1]
        return atten_out

    def top_k_accuracy(self, labels, predict_label,k=1):
        batch_size = labels.get_shape().as_list()[0]
        #print('self.predict_label',self.predict_label)
        right_1 = tf.to_float(tf.nn.in_top_k(predictions=predict_label, targets=labels, k=k))          #find the max value of prediction
        num_correct_per_batch = tf.reduce_sum(right_1)
        acc = num_correct_per_batch / batch_size

        tf.summary.scalar('acc', acc)

        return acc

    def top_k_accuracy_f(self, labels, predict_label_f, k=1):
        batch_size = labels.get_shape().as_list()[0]
        #print('self.predict_label_f',self.predict_label_f)
        right_1 = tf.to_float(tf.nn.in_top_k(predictions=predict_label_f, targets=labels, k=k))          #find the max value of prediction
        num_correct_per_batch = tf.reduce_sum(right_1)
        acc = num_correct_per_batch / batch_size

        tf.summary.scalar('acc_f', acc)

        return acc
    def top_k_accuracy_validation(self, labels, k=1):
        batch_size = labels.get_shape().as_list()[0]
        right_1 = tf.to_float(tf.nn.in_top_k(predictions=self.predict_label, targets=labels, k=k))
        num_correct_per_batch = tf.reduce_sum(right_1)
        acc = num_correct_per_batch / batch_size

        return acc

    def top_k_accuracy_validation_f(self, labels, k=1):
        batch_size = labels.get_shape().as_list()[0]
        right_1 = tf.to_float(tf.nn.in_top_k(predictions=self.predict_label_f, targets=labels, k=k))
        num_correct_per_batch = tf.reduce_sum(right_1)
        acc = num_correct_per_batch / batch_size

        return acc

    def predict_inference(self, x, salmask_input, salmask_feature,
                          reuse=False):  # x size at 224*224*3,(-1,1) , salmask(0,1)

        x = self.conv_mask(x, salmask_input)

        self.mask_input = x
        conv_1 = self.conv('1conv_1', x, ksize=7, filters_out=64, stride=2, reuse=reuse)
        conv_1 = self.conv_mask(conv_1, salmask_feature)

        # feature size at 56*56*64
        max_pool_1 = self.max_pool(conv_1, ksize=3, stride=2)
        block_1_1 = self.vv_building_block('1block_1_1', max_pool_1, output_channel=64, stride=1, reuse=reuse)
        block_1_1 = self.conv_mask(block_1_1, salmask_feature)

        # feature size at  28*28*128
        block_2_1 = self.vv_building_block('1block_2_1', block_1_1, output_channel=128, stride=2, reuse=reuse)
        block_2_1 = self.conv_mask(block_2_1, salmask_feature)

        # # feature size at 14*14*128
        block_3_1 = self.vv_building_block('1block_3_1', block_2_1, output_channel=128, stride=2, reuse=reuse)
        block_3_1 = self.conv_mask(block_3_1, salmask_feature)

        # feature size at 7*7*256
        block_4_1 = self.vv_building_block('1block_4_1', block_3_1, output_channel=256, stride=2, reuse=reuse)

        ave_pool_1 = tf.reduce_mean(block_4_1, reduction_indices=[1, 2], name="avg_pool")  # feature size at 1*1*512
        fc_1 = self.fc('fc_11', ave_pool_1, class_num=256, flat=False, reuse=reuse)  # feature size at 1024
        fc_2 = self.fc('fc_22', fc_1, class_num=128, flat=False, reuse=reuse)
        self.fc2 = fc_2
        fc_3 = self.fc('fc_33', fc_2, class_num=2, flat=False, linear=True, reuse=reuse)

        return block_4_1  # ave_pool_1

    def polar_inference(self, x, reuse=False):
        conv_1 = self.conv('1conv_1', x, ksize=7, filters_out=32, stride=2, reuse=reuse)

        # feature size at 56*56*64
        max_pool_1 = self.max_pool(conv_1, ksize=3, stride=2)
        block_1_1 = self.vv_building_block('1block_1_1', max_pool_1, output_channel=32, stride=1, reuse=reuse)

        # feature size at  28*28*128
        block_2_1 = self.vv_building_block('1block_2_1', block_1_1, output_channel=32, stride=2, reuse=reuse)

        # # feature size at 14*14*128
        block_3_1 = self.vv_building_block('1block_3_1', block_2_1, output_channel=64, stride=2, reuse=reuse)

        # feature size at 7*7*256
        block_4_1 = self.vv_building_block('1block_4_1', block_3_1, output_channel=64, stride=2, reuse=reuse)

        return block_4_1  # ave_pool_1


    def get_feature_inference(self, x, reuse= None):     # x size at 224*224*3
        #with tf.variable_scope('scale_1'):  # feature size at 112*112*64
        conv_1=self.conv('conv_1', x, ksize=7, filters_out=64, stride=2, reuse=reuse)       #1000: 500*500*64
        print('1', conv_1)
        #with tf.variable_scope('scale_2'):  # feature size at 56*56*64
        max_pool_1= self.max_pool(conv_1, ksize=3, stride=2)
        block_1_1 = self.v_building_block('block_1_1', max_pool_1, output_channel=64, stride=1, reuse= reuse)
        block_1_2 = self.v_building_block('block_1_2', block_1_1, output_channel=64, stride=1, reuse= reuse)         #56*56*64      #1000:250*250*64
        print('2', block_1_2)
        #with tf.variable_scope('scale_3'):  # feature size at  28*28*128
        block_2_1 = self.v_building_block('block_2_1', block_1_2, output_channel=128, stride=2, reuse= reuse)
        block_2_2 = self.v_building_block('block_2_2', block_2_1, output_channel=128, stride=1, reuse= reuse)         #28*28*128      #1000:125, 125, 128
        print('3',block_2_2)
        #with tf.variable_scope('scale_4'):  # feature size at 14*14*256
        block_3_1 = self.v_building_block('block_3_1', block_2_2, output_channel=256, stride=2, reuse= reuse)
        block_3_2 = self.v_building_block('block_3_2', block_3_1, output_channel=256, stride=1, reuse= reuse)         #14*14*256        #1000:63, 63, 256
        print('4', block_3_2)
        #with tf.variable_scope('scale_5'):  # feature size at 7*7*512
        block_4_1 = self.v_building_block('block_4_1', block_3_2, output_channel=512, stride=2, reuse= reuse)
        block_4_2 = self.v_building_block('block_4_2', block_4_1, output_channel=512, stride=1, reuse= reuse)         #7*7*512      #1000:32, 32, 512
        print('5', block_4_2)
        # #with tf.variable_scope('scale_6'):  # feature size flap to 2 dim
        # ave_pool_1 = tf.reduce_mean(block_4_2, reduction_indices=[1, 2], name="avg_pool")   # feature size at 1*1*512
        # #fc_1 = self.fc('fc_1', ave_pool_1, class_num= 256, flat=True)   # feature size at 1024
        # fc_2 = self.fc('fc_2', ave_pool_1, class_num=1024, flat=False)
        # fc_3 = self.fc('fc_3', fc_2, class_num=735, flat=False, linear=True)

        # high_feature = tf.reshape(fc_3, [fc_3.get_shape()[0].value, self.cell_size, self.cell_size, -1])    #[batch, 7, 7, 30]
        mid_conv_1 = self.conv('mid_conv_1', block_1_2, ksize=1, filters_out=128, stride=1, reuse= reuse)     #56*56*128
        mid_conv_2 = self.conv('mid_conv_2', block_2_2, ksize=1, filters_out=128, stride=1, reuse= reuse)     #28*28*128
        mid_conv_3 = self.conv('mid_conv_3', block_3_2, ksize=1, filters_out=128, stride=1, reuse= reuse)     #14*14*128
        mid_conv_4 = self.conv('mid_conv_4', block_4_2, ksize=1, filters_out=128, stride=1, reuse= reuse)     #7*7*128

        tempsize = mid_conv_2.get_shape().as_list()
        fin_conv_1 = tf.image.resize_images(mid_conv_1, [tempsize[1], tempsize[2]])     #28*28*128      #1000:125, 125, 128
        #print('mul',fin_conv_1)
        fin_conv_2 = tf.image.resize_images(mid_conv_2, [tempsize[1], tempsize[2]])     #28*28*128
        fin_conv_3 = tf.image.resize_images(mid_conv_3, [tempsize[1], tempsize[2]])     #28*28*128
        fin_conv_4 = tf.image.resize_images(mid_conv_4, [tempsize[1], tempsize[2]])     #28*28*128
        # fin_high_feature = tf.image.resize_images(high_feature, [tempsize[1], tempsize[2]])     #28*28*30

        FeatureMap = tf.concat([fin_conv_1, fin_conv_2, fin_conv_3, fin_conv_4], axis=3)     #28*28, fin_high_feature

        return FeatureMap


    def SalencyMap(self, FeatureMap, reuse=None):
        conv_2 =  self.conv('conv_2', FeatureMap, ksize=3, filters_out=256, stride=1, reuse=reuse)       #28*28*512         #125,125,512
        conv_3 =  self.conv('conv_3', conv_2, ksize=1, filters_out=128, stride=1, reuse=reuse)       #28*28*256
        conv_4 =  self.conv('conv_4', conv_3, ksize=3, filters_out=64, stride=1, reuse=reuse)       #28*28*128
        conv_5 =  self.conv('conv_5', conv_4, ksize=1, filters_out=64, stride=1, reuse=reuse)       #28*28*64
        dconv_1 = self.dconv('dconv_1', conv_5, ksize=2, filters_out=16, stride=2, reuse=reuse)       #56*56*16         #250,250
        dconv_2 = self.dconv('dconv_2', dconv_1, ksize=2, filters_out=1, stride=2, liner=True, reuse=reuse)      #112*112*1     #500,500
        dconv_2 = tf.image.resize_images(dconv_2, [112, 112])
        #print('sal',dconv_2)

        return dconv_2


    def _loss(self, labGT, salGT):  #
        batch_size = labGT.get_shape().as_list()[0]
        weight_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=None)
        loss_weight = tf.add_n(weight_loss)

        norm_predict_salmap = self._normlized(self.predict_salmap)     #self._normlized
        norm_GT = self._normlized(salGT)   #self._normlized
        a = norm_GT * tf.log(self.eps + norm_GT / (norm_predict_salmap +self.eps))
        # print(a.get_shape().as_list())
        loss_salency = tf.reduce_sum(a)

        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict_label, labels=labGT)        #labels is in form of 1 dim,[batch_size], logistics is in 2 dims, [batch_size, output_num]
        # print(ce.get_shape().as_list())
        loss_label = tf.reduce_sum(ce)

        self.loss_label = loss_label / batch_size
        self.loss_salency = loss_salency / batch_size
        self.loss_weight = loss_weight
        loss = self.loss_weight + self.beta1 * self.loss_label + self.beta2 * self.loss_salency
        self.loss = loss

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('loss_salency', self.loss_salency)
        tf.summary.scalar('loss_label', self.loss_label)
        tf.summary.scalar('loss_weight', self.loss_weight)
        return self.loss, self.loss_label, self.loss_weight, self.loss_salency



    def _loss_double_mask(self, labGT, salGT, predict_salmap, predict_label,predict_label_f,  beta1, beta2, beta3):  #
        batch_size = labGT.get_shape().as_list()[0]
        weight_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=None)
        loss_weight = tf.add_n(weight_loss)

        norm_predict_salmap = self._normlized(predict_salmap)     #self._normlized
        norm_GT = self._normlized(salGT)   #self._normlized
        # a = norm_GT * tf.log(self.eps + norm_GT / (norm_predict_salmap +self.eps))

        a = norm_predict_salmap * tf.log(self.eps + norm_predict_salmap / (norm_GT + self.eps))
        loss_salency = tf.reduce_sum(a)

        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_label, labels=labGT)        #labels is in form of 1 dim,[batch_size], logistics is in 2 dims, [batch_size, output_num]
        loss_label = tf.reduce_sum(ce)

        ce_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_label_f, labels=labGT)        #labels is in form of 1 dim,[batch_size], logistics is in 2 dims, [batch_size, output_num]
        loss_label_f = tf.reduce_sum(ce_1)

        self.loss_label = loss_label / batch_size
        self.loss_label_f = loss_label_f / batch_size
        self.loss_salency = loss_salency / batch_size
        self.loss_weight = loss_weight
        loss = self.loss_weight + beta1 * self.loss_label + beta3 * self.loss_salency + beta2 * self.loss_label_f
        self.loss = loss

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('loss_salency', self.loss_salency)
        tf.summary.scalar('loss_label', self.loss_label)
        tf.summary.scalar('loss_label_f', self.loss_label_f)
        tf.summary.scalar('loss_weight', self.loss_weight)

        return loss, self.loss_label, self.loss_label_f, self.loss_weight, self.loss_salency


    def _loss_semi(self, labGT, predict_salmap, predict_salmap_1):  #salGT,
        batch_size = labGT.get_shape().as_list()[0]
        weight_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=None)
        loss_weight = tf.add_n(weight_loss)

        norm_predict_salmap = self._normlized(predict_salmap)     #self._normlized
        norm_GT = self._normlized(salGT)
        a = norm_GT * tf.log(self.eps + norm_GT / (norm_predict_salmap + self.eps))
        # print(a.get_shape().as_list())
        loss_salency = tf.reduce_sum(a)


        norm_predict_salmap_1 = self._normlized(predict_salmap_1)

        b = norm_predict_salmap * tf.log(self.eps + norm_predict_salmap / (norm_predict_salmap_1 + self.eps))
        # print(a.get_shape().as_list())
        loss_semi = tf.reduce_sum(b)

        #loss_salency = tf.reduce_sum(a)

        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict_label, labels=labGT)        #labels is in form of 1 dim,[batch_size], logistics is in 2 dims, [batch_size, output_num]
        # print(ce.get_shape().as_list())
        loss_label = tf.reduce_sum(ce)

        self.loss_salency = loss_salency / batch_size
        self.loss_label = loss_label / batch_size
        self.loss_semi = loss_semi / batch_size
        self.loss_weight = loss_weight

        #self.test=self.beta2
        loss = self.loss_weight + self.beta1 * self.loss_label + self.beta2 * self.loss_salency + self.beta3 * self.loss_semi
        self.loss = loss
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('loss_semi', self.loss_semi)
        tf.summary.scalar('loss_label', self.loss_label)
        tf.summary.scalar('loss_weight', self.loss_weight)
        tf.summary.scalar('loss_salency', self.loss_salency)
        return self.loss,  self.loss_label, self.loss_weight, self.loss_salency, self.loss_semi


    def only_loss_semi(self, predict_salmap, predict_salmap_change):
        norm_predict_salmap = self._normlized(predict_salmap)
        norm_predict_salmap_1 = self._normlized(predict_salmap_change)
        b = norm_predict_salmap * tf.log(self.eps + norm_predict_salmap / (norm_predict_salmap_1 + self.eps))
        # print(a.get_shape().as_list())
        loss_semi = tf.reduce_sum(b)
        return loss_semi#,norm_predict_salmap,norm_predict_salmap_1


    def _loss_val(self, labGT, salGT):
        batch_size = labGT.get_shape().as_list()[0]
        weight_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=None)
        loss_weight = tf.add_n(weight_loss)

        norm_predict_salmap = self._normlized(self.predict_salmap)     #
        norm_GT = self._normlized(salGT)    #self._normlized
        loss_salency = tf.reduce_sum(norm_GT * tf.log(self.eps + norm_GT / (norm_predict_salmap +self.eps)))

        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict_label, labels=labGT)        #labels is in form of 1 dim,[batch_size], logistics is in 2 dims, [batch_size, output_num]
        loss_label = tf.reduce_sum(ce)

        self.loss_label = loss_label / batch_size
        self.loss_salency = loss_salency / batch_size
        loss = self.loss_weight + self.beta1 * self.loss_label + self.beta2 * self.loss_salency
        self.loss = loss

        return self.loss, self.loss_label, self.loss_salency

    def _loss_double_mask_val(self, labGT, salGT, predict_salmap, predict_label,predict_label_f, beta1, beta2, beta3):
        batch_size = labGT.get_shape().as_list()[0]
        weight_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=None)
        loss_weight = tf.add_n(weight_loss)

        norm_predict_salmap = self._normlized(predict_salmap)     #
        norm_GT = self._normlized(salGT)    #self._normlized
        loss_salency = tf.reduce_sum(norm_GT * tf.log(self.eps + norm_GT / (norm_predict_salmap +self.eps)))

        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_label, labels=labGT)        #labels is in form of 1 dim,[batch_size], logistics is in 2 dims, [batch_size, output_num]
        loss_label = tf.reduce_sum(ce)

        ce_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_label_f, labels=labGT)        #labels is in form of 1 dim,[batch_size], logistics is in 2 dims, [batch_size, output_num]
        # print(ce.get_shape().as_list())
        loss_label_f = tf.reduce_sum(ce_1)

        loss_label = loss_label / batch_size
        loss_label_f = loss_label_f / batch_size
        loss_salency = loss_salency / batch_size
        loss = loss_weight + beta1 * loss_label + beta2 * loss_label_f + beta3 * loss_salency


        return loss, loss_label, loss_label_f, loss_salency


    def _only_loss_label(self, labGT):
        batch_size = labGT.get_shape().as_list()[0]

        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict_label, labels=labGT)        #labels is in form of 1 dim,[batch_size], logistics is in 2 dims, [batch_size, output_num]
        loss_label = tf.reduce_mean(ce)     # 改为 mean

        loss_label = loss_label / batch_size

        return loss_label

    def _train(self, loss):

        opt = tf.train.AdamOptimizer(self.init_learning_rate, beta1=0.9, beta2=0.999, epsilon= 1e-08)

        gradients = opt.compute_gradients(loss)    #all variables are trainable
        apply_gradient_op = opt.apply_gradients(gradients)      #, global_step=self.global_step
        self.train = apply_gradient_op

        return apply_gradient_op


    def _grad_cam(self, loss_label, layer_output):
        grads = tf.gradients(loss_label, layer_output)[0]
        norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

        return norm_grads


    def _grad_cam_1(self,  layer_output):
        grads_0 = tf.gradients(self.predict_label[:,0], layer_output)[0]   #不要取一个batch吧,这个[0]不是取某batch中的某一维，而是从[tensorxxx]中取出 tensor

        norm_grads_0 = tf.div(grads_0, (tf.sqrt(tf.reduce_mean(tf.square(grads_0))) + tf.constant(1e-5)))

        grads_1 = tf.gradients(self.predict_label[:,1], layer_output)[0]
        norm_grads_1 = tf.div(grads_1, tf.sqrt(tf.reduce_mean(tf.square(grads_1))) + tf.constant(1e-5))
        #print(norm_grads_1)

        return norm_grads_0, norm_grads_1

    def _grad_bp(self, feature, input, g):
        with g.gradient_override_map({'Relu': 'GuidedRelu'}):
            #grads = tf.gradients(self.predict_label, input)[0]
            grads = tf.gradients(feature, input)[0]  #输出input的size  [batch,w,h,3]

            #grads = tf.gradients(self.predict_label[:, 1], input)[0]
        return grads

    def _gradient_norm(self, grads):

        return



    # def _train_semi(self):
    #
    #     opt = tf.train.AdamOptimizer(self.init_learning_rate, beta1=0.9, beta2=0.999, epsilon= 1e-08)
    #
    #     gradients = opt.compute_gradients(self.loss_semi)    #all variables are trainable
    #     apply_gradient_op = opt.apply_gradients(gradients)      #, global_step=self.global_step
    #     self.train_semi = apply_gradient_op
    #
    #     return apply_gradient_op