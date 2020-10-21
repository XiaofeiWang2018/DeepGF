
import BasicNet
import tensorflow as tf
from utils import *
import CNN_attention_combine_slim_multi_scale_33 as CVPR_Network

class Net(BasicNet.BasicNet):

    def __init__(self, batch_size=4, n_steps=5, n_layers=2, n_neurons=256, n_outputs=2,
                 init_lr=10 ** (-4), #max_step=15,
    ):
        super(Net, self).__init__()

        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        #self.max_step = max_step

        self.init_lr = init_lr

        self.predict_label = []
        self.loss = []
        self.loss_weight = []
        self.train = []

    def inference_model_1(self, x, is_training=True):
        """
        CNN+Convlstm+CNN+fc
        """
        cnn_sequential = self.multi_CNNoutput_2D('cnn_mul', x,
                                                 is_training)  # cnn_sequential = [batch, max_step, n_lstm_input]
        lstm_input = cnn_sequential

        fc_output= self.convlstm_cnn_fc('lstm1', lstm_input, n_layers=self.n_layers,is_training=is_training) #[batch, n_steps, 28,28,64]

        self.predict_label = fc_output
        return fc_output



    def inference_model_2(self, x, is_training=True):         # x= [batch, n_step, w, h, c]
        """
        pretrained_Alex+lstm+fc
        """
        cnn_sequential = self.multi_CNNoutput_1D('cnn_mul', x, is_training)     #cnn_sequential = [batch, max_step, n_lstm_input]
        lstm_input = cnn_sequential
        rnn_output, fc_output, states = self.lstm_fc('lstm2', lstm_input, n_layers=self.n_layers,
                                                  n_neurons=self.n_neurons, n_outputs=self.n_outputs,
                                                  sequence_length=self.n_steps)
        self.predict_label = fc_output
        return fc_output

    def inference_model_3(self, x,is_training=True):         # x= [batch, n_step, w, h, c]
        """
        pretrained_Alex+Convlstm+GAP+fc
        """
        cnn_sequential = self.multi_CNNoutput_pretrainedAlex(x,is_training=is_training)


        lstm_input = cnn_sequential
        fc_output = self.convlstm_gap_fc('lstm3', lstm_input, n_layers=self.n_layers,
                                         is_training=is_training)  # [batch, n_steps, 28,28,64]

        self.predict_label = fc_output
        return fc_output

    def inference_model_4(self, x,label_attention_map,is_training=True):         # x= [batch, n_step, w, h, c]
        """
        pretrained_CVPR+Convlstm+GAP+fc+atten
        """
        cnn_sequential = self.multi_CNNoutput_pretrainedCVPR(x,label_attention_map,is_training=is_training)


        lstm_input = cnn_sequential
        fc_output = self.convlstm_gap_fc('lstm4', lstm_input, n_layers=self.n_layers,
                                         is_training=is_training)  # [batch, n_steps, 28,28,64]

        self.predict_label = fc_output
        return fc_output

    def inference_model_5(self, x,label_attention_map,label_polar_map,is_training=True):         # x= [batch, n_step, w, h, c]
        """
        pretrained_CVPR+Convlstm+GAP+fc+atten+polar
        """

        cnn_sequential = self.multi_CNNoutput_pretrainedCVPR(x,label_attention_map,is_training=is_training)

        cnn_sequential_polar=self.multi_CNNoutput_polar('polar',x,label_polar_map,is_training=is_training)
        lstm_input = tf.concat([cnn_sequential,cnn_sequential_polar],axis=4)
        fc_output = self.convlstm_gap_fc('lstm5', lstm_input, n_layers=self.n_layers,
                                         is_training=is_training)  # [batch, n_steps, 28,28,64]

        self.predict_label = fc_output
        return fc_output



    def inference_model_6(self, x,label_attention_map,label_polar_map,delta_year,init_a,init_b,is_training=True):         # x= [batch, n_step, w, h, c]
        """
       pretrained_CVPR+Convlstm+GAP+fc+atten+timegate
        """

        cnn_sequential = self.multi_CNNoutput_pretrainedCVPR(x,label_attention_map,is_training=is_training)

        cnn_sequential_polar=self.multi_CNNoutput_polar('polar',x,label_polar_map,is_training=is_training)
        lstm_input = tf.concat([cnn_sequential,cnn_sequential_polar],axis=4)
        # lstm_input = cnn_sequential
        fc_output = self.convlstm_gap_fc_timegate('lstm6', lstm_input, n_layers=self.n_layers,
                                         is_training=is_training,delta_year=delta_year,init_a=init_a,init_b=init_b)  # [batch, n_steps, 28,28,64]

        self.predict_label = fc_output
        return fc_output

    def inference_model_7(self, x,label_attention_map,label_polar_map,delta_year,init_a,init_b,is_training=True):
        """
       pretrained_CVPR+Convlstm+GAP+fc+atten+timegate
        """

        cnn_sequential = self.multi_CNNoutput_pretrainedCVPR(x,label_attention_map,is_training=is_training)

        lstm_input = cnn_sequential  # 7x7x256
        fc_output = self.convlstm_gap_fc_timegate('lstm7', lstm_input, n_layers=self.n_layers,
                                         is_training=is_training,delta_year=delta_year,init_a=init_a,init_b=init_b)  # [batch, n_steps, 28,28,64]

        self.predict_label = fc_output
        return fc_output

    def inference_model_8(self, x,label_attention_map,label_polar_map,delta_year,init_a,init_b,is_training=True):         # x= [batch, n_step, w, h, c]
        """
       CVPR(polar)+Lstm(timegate)+GAP+fc
        """

        cnn_sequential = self.multi_CNNoutput_pretrainedCVPR_polar(x,label_attention_map,label_polar_map,is_training=is_training)

        lstm_input = tf.reduce_mean(cnn_sequential,axis=(2,3)) #[batch_size,n_steps,256+64]

        fc_output,a_1,b_1,a_2,b_2 = self.lstm_gap_fc_timegate('lstm8', lstm_input, n_layers=self.n_layers,
                                         is_training=is_training,delta_year=delta_year,init_a=init_a,init_b=init_b)  # [batch, n_steps, 28,28,64]

        self.predict_label = fc_output
        return fc_output ,a_1,b_1,a_2,b_2

    def CNN_1D(self, scope_name, x, is_training, reuse=True):       #[batch, w, h, c]
        with tf.variable_scope(scope_name):
            conv_1 = self.conv('conv_1', x, ksize=7, filters_out=32, stride=2, is_training=is_training, batch_norm=True, reuse=reuse)
            norm_1 = tf.nn.local_response_normalization(conv_1)
            pool_1 = self.max_pool(norm_1, ksize=3, stride=2)

            conv_2 = self.conv('conv_2', pool_1, ksize=3, filters_out=64, stride=1, is_training=is_training, batch_norm=True, reuse=reuse)
            norm_2 = tf.nn.local_response_normalization(conv_2)
            pool_2 = self.max_pool(norm_2, ksize=3, stride=2)

            conv_3 = self.conv('conv_3', pool_2, ksize=3, filters_out=128, stride=1, is_training=is_training, batch_norm=True, reuse=reuse)
            pool_3 = self.max_pool(conv_3, ksize=3, stride=2)

            conv_4 = self.conv('conv_4', pool_3, ksize=3, filters_out=128, stride=1, is_training=is_training, batch_norm=True, reuse=reuse)
            pool_4 = self.max_pool(conv_4, ksize=3, stride=2)

            # with tf.variable_scope('scale_6'):  # feature size flap to 2 dim
            ave_pool_1 = tf.reduce_mean(pool_4, reduction_indices=[1, 2], name="avg_pool")  # feature size at batch*1*1*512
            n_input = ave_pool_1.get_shape().as_list()[-1]
            output = tf.reshape(ave_pool_1, [self.batch_size, n_input])

            return output

    def CNN_2D(self, scope_name, x, is_training, reuse=True):
        with tf.variable_scope(scope_name):
            conv_1 = self.conv('conv_1', x, ksize=7, filters_out=32, stride=2, is_training=is_training, batch_norm=True, reuse=reuse)
            norm_1 = tf.nn.local_response_normalization(conv_1)
            pool_1 = self.max_pool(norm_1, ksize=3, stride=2)

            conv_2 = self.conv('conv_2', pool_1, ksize=3, filters_out=64, stride=1, is_training=is_training, batch_norm=True, reuse=reuse)
            norm_2 = tf.nn.local_response_normalization(conv_2)
            pool_2 = self.max_pool(norm_2, ksize=3, stride=2)


            return pool_2

    def CNN_polar(self, scope_name, x, is_training, reuse=True):
        with tf.variable_scope(scope_name):
            conv_1 = self.conv('conv_1', x, ksize=7, filters_out=32, stride=2, is_training=is_training,batch_norm=True, reuse=reuse)
            norm_1 = tf.nn.local_response_normalization(conv_1)
            pool_1 = self.max_pool(norm_1, ksize=3, stride=2)

            conv_2 = self.conv('conv_2', pool_1, ksize=3, filters_out=64, stride=2, is_training=is_training,
                               batch_norm=True, reuse=reuse)
            norm_2 = tf.nn.local_response_normalization(conv_2)
            pool_2 = self.max_pool(norm_2, ksize=3, stride=2)

            conv_3 = self.conv('conv_3', pool_2, ksize=3, filters_out=128, stride=1, is_training=is_training,
                               batch_norm=True, reuse=reuse)
            norm_3 = tf.nn.local_response_normalization(conv_3)
            pool_3 = self.max_pool(norm_3, ksize=3, stride=2)

            return pool_3

    def multi_CNNoutput_1D(self, scope_name, x, is_training):     #x_size = [batch, n_step, w, h, c], sequential
        with tf.variable_scope(scope_name):
            # x_shape define in placeholder as None, thus we cannot use x.get_shape().as_list()[1] as n_steps
            n_steps =x.get_shape().as_list()[1]
            for i in range(n_steps):
                x_0 = x[:, i, :, :, :]
                if i == 0:
                    y_0 = self.CNN_1D('CNN_1', x_0, is_training, reuse=False)  # [batch, n_input]
                    y_0 = tf.expand_dims(y_0, 1)
                    y = y_0
                else:
                    y_0 = self.CNN_1D('CNN_1', x_0, is_training, reuse=True)  # [batch, n_input]
                    y_0 = tf.expand_dims(y_0, 1)
                    y = tf.concat([y,y_0], 1)

        return y        #y_size = [batch, n_step, n_feature, c]

    def multi_CNNoutput_2D(self, scope_name, x, is_training):     #x_size = [batch, n_step, w, h, c], sequential
        with tf.variable_scope(scope_name):
            n_steps =x.get_shape().as_list()[1]
            for i in range(n_steps):
                x_0 = x[:, i, :, :, :]
                if i == 0:
                    y_0 = self.CNN_2D('CNN_2', x_0, is_training, reuse=False)  # [batch, n_input]
                    y_0 = tf.expand_dims(y_0, 1)
                    y = y_0
                else:
                    y_0 = self.CNN_2D('CNN_2', x_0, is_training, reuse=True)  # [batch, n_input]
                    y_0 = tf.expand_dims(y_0, 1)
                    y = tf.concat([y,y_0], 1)
        return y

    def multi_CNNoutput_pretrainedAlex(self, x, is_training):
        net = AlexNet(x, num_classes=2, is_training=is_training, dropout_keep_prob=0.8)
        n_steps =x.get_shape().as_list()[1]
        for i in range(n_steps):
            x_0 = x[:, i, :, :, :]
            if i == 0:
                with slim.arg_scope(AlexNet.alexnet_v2_arg_scope(reuse=False)):
                    y_0 = net.alexnet_5x5_complex(x_0)[7]  # [batch, n_input]
                y_0 = tf.expand_dims(y_0, 1)
                y = y_0
            else:
                with slim.arg_scope(AlexNet.alexnet_v2_arg_scope(reuse=True)):
                    y_0 = net.alexnet_5x5_complex(x_0)[7]
                y_0 = tf.expand_dims(y_0, 1)
                y = tf.concat([y,y_0], 1)

        return y

    def multi_CNNoutput_pretrainedCVPR_polar(self, x, label_attention_map,polar,is_training):
        net = CVPR_Network.Net()
        net.is_training = True
        n_steps =x.get_shape().as_list()[1]
        for i in range(n_steps):
            x_0 = x[:, i, :, :, :]
            label_attention_map_0= label_attention_map[:, i, :, :, :]
            polar_0=polar[:, i, :, :, :]
            if i == 0:
                y_0=net.get_label_attention_polar_inference(x_0, label_attention_map_0,polar_0,reuse=False)
                y_0 = tf.expand_dims(y_0, 1)
                y = y_0
            else:
                y_0=net.get_label_attention_polar_inference(x_0, label_attention_map_0,polar_0,reuse=True)
                y_0 = tf.expand_dims(y_0, 1)
                y = tf.concat([y,y_0], 1)

        return y
    def multi_CNNoutput_pretrainedCVPR(self, x, label_attention_map,is_training):
        net = CVPR_Network.Net()
        net.is_training = True
        n_steps =x.get_shape().as_list()[1]
        for i in range(n_steps):
            x_0 = x[:, i, :, :, :]
            label_attention_map_0= label_attention_map[:, i, :, :, :]
            if i == 0:
                y_0=net.get_label_attention_inference(x_0, label_attention_map_0,reuse=False)
                y_0 = tf.expand_dims(y_0, 1)
                y = y_0
            else:
                y_0=net.get_label_attention_inference(x_0, label_attention_map_0,reuse=True)
                y_0 = tf.expand_dims(y_0, 1)
                y = tf.concat([y,y_0], 1)

        return y

    def multi_CNNoutput_polar(self, scope_name,x, label_attention_map, is_training):
        with tf.variable_scope(scope_name):
            n_steps =x.get_shape().as_list()[1]
            for i in range(n_steps):
                label_attention_map_0 = label_attention_map[:, i, :, :, :]
                if i == 0:
                    y_0 = self.CNN_polar('CNN_2', label_attention_map_0, is_training, reuse=False)  # [batch, n_input]
                    y_0 = tf.expand_dims(y_0, 1)
                    y = y_0
                else:
                    y_0 = self.CNN_polar('CNN_2', label_attention_map_0, is_training, reuse=True)  # [batch, n_input]
                    y_0 = tf.expand_dims(y_0, 1)
                    y = tf.concat([y,y_0], 1)
            return y


    def _loss(self, predict_label, labGT):     #label_predict_op=(batch, n_steps, n_outputs)  label=[batch, n_steps]
        weight_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=None)
        loss_weight = tf.add_n(weight_loss)

        n_steps = labGT.get_shape().as_list()[1]
        labGT=tf.reshape(labGT,shape=[labGT.get_shape().as_list()[0]*labGT.get_shape().as_list()[1]])
        predict_label=tf.reshape(predict_label,shape=[predict_label.get_shape().as_list()[0]*predict_label.get_shape().as_list()[1],predict_label.get_shape().as_list()[2]])


        loss_label = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labGT, logits=predict_label))

        loss = loss_weight + loss_label #+ loss_direction
        self.loss = loss


        return loss, loss_label, loss_weight

    def _loss_liliu(self, predict_label, labGT):  # label_predict_op=(batch, n_steps, n_outputs)  label=[batch, n_steps]
        weight_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=None)
        loss_weight = tf.add_n(weight_loss)


        batch_size = labGT.get_shape().as_list()[0]
        n_step = labGT.get_shape().as_list()[1]
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_label, labels=labGT)

        loss_label = tf.reduce_sum(ce) / (batch_size * n_step)

        # loss_p

        loss = loss_weight + loss_label  # + loss_direction
        self.loss = loss
        return loss, loss_label, loss_weight





    def _loss_per_batch(self, predict_label, labGT):  # label_predict_op=(batch, n_steps, n_outputs)  label=[batch, n_steps]

        ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_label, labels=labGT) #[batch_size,n_steps]

        loss_label = tf.reduce_mean(ce,axis=1)

        return  loss_label


    def _loss_val(self, predict_label, label):
        """

        :param predict_label: batch_size,n_steps,2
        :param label: batch_size,n_steps
        :return:
        """
        batch_size = label.get_shape().as_list()[0]
        n_steps = label.get_shape().as_list()[1]
        loss_label = 0
        """loss_label"""
        for i in range(n_steps):
            loss_step = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label[:, i], logits=predict_label[:, i, :]))
            loss_label = loss_label + loss_step

        loss_label = loss_label / n_steps

        """loss_weight"""
        weight_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=None)
        loss_weight = tf.add_n(weight_loss)


        loss_val = loss_weight + loss_label

        return loss_val

    def _train(self):

        opt = tf.train.AdamOptimizer(self.init_lr, beta1=0.9, beta2=0.999, epsilon= 1e-08)

        gradients = opt.compute_gradients(self.loss)    #all variables are trainable
        apply_gradient_op = opt.apply_gradients(gradients)      #, global_step=self.global_step
        self.train = apply_gradient_op
        return apply_gradient_op




    def _top_k_accuracy(self, predict_label, labels, k=1):

        batch_size = labels.get_shape().as_list()[0]
        n_steps = labels.get_shape().as_list()[1]
        acc_all = 0
        for i in range(n_steps):
            right_1 = tf.to_float(tf.nn.in_top_k(predictions=predict_label[:, i, :], targets=labels[:, i], k=k))
            num_correct_per_batch = tf.reduce_sum(right_1)
            acc = num_correct_per_batch / batch_size
            acc_all = acc_all + acc


        return acc_all/n_steps

# if __name__ == '__main__':
