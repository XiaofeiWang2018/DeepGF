
import tensorflow as tf
import math
# import numpy as np
# from config import Config
from ConvLSTMCell_timegate import ConvLSTMCell_timegate
from  LSTMCell_timegate import LSTMCell_timegate





class BasicNet(object):
    weight_decay = 5*1e-6
    bias_conv_init = 0.1 #weight init for biasis
    bias_fc_init = 0.1
    leaky_alpha = 0.1
    is_training = False
    #LL_VARIABLES = 'll_variables'

    #batch_normalization = True
    # class_num = 2

    def _get_variable(self,
                      name,
                      shape,
                      initializer,
                      weight_decay= weight_decay,
                      dtype='float32',
                      trainable=True, AAAI_VARIABLES=None):  # pretrain/ initial/

        if weight_decay >0:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None

        collection = [tf.GraphKeys.GLOBAL_VARIABLES]  #, LL_VARIABLES

        return tf.get_variable(name= name,
                               shape= shape,
                               initializer= initializer,
                               regularizer=regularizer,
                               collections= collection,
                               dtype= dtype,
                               trainable= trainable,
                               )

    def conv(self, scope_name, x, ksize, filters_out, stride=1, batch_norm= True, is_training= True, liner = False, reuse=None):
        with tf.variable_scope(scope_name, reuse=reuse):
            filters_in = x.get_shape()[-1].value

            shape = [ksize, ksize, filters_in, filters_out]  # conv kernel size
            weights = self._get_variable('weights',
                                    shape=shape,
                                    initializer=tf.contrib.layers.xavier_initializer()  # need to set seed number
                                    )
            bias = self._get_variable('bias',
                                 shape=[filters_out],
                                 initializer=tf.constant_initializer(self.bias_conv_init)
                                 )
            conv = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
            conv_bias = tf.nn.bias_add(conv, bias, name='linearout')

            if batch_norm:
                out = self.bn(conv_bias, is_training=is_training)
            else:
                out = conv_bias

            if liner:
                return out
            else:
                return self.leaky_relu(out)

    def dconv(self, scope_name, x, ksize, filters_out, stride=1, liner=False):
        with tf.variable_scope(scope_name):
            filters_in = x.get_shape()[-1].value

            shape = [ksize, ksize, filters_out, filters_in]
            weights = self._get_variable('weights',
                                    shape=shape,
                                    initializer=tf.contrib.layers.xavier_initializer()
                                    )
            bias = self._get_variable('bias',
                                 shape=[filters_out],
                                 initializer=tf.constant_initializer(self.bias_conv_init)
                                 )

            output_shape = tf.stack([tf.shape(x)[0], tf.shape(x)[1] * stride, tf.shape(x)[2] * stride, filters_out])
            conv = tf.nn.conv2d_transpose(x, weights, output_shape, strides=[1, stride, stride, 1], padding='SAME')
            conv_biased = tf.nn.bias_add(conv, bias, name='linearout')

            if liner:
                return conv_biased
            else:
                return self.leaky_relu(conv_biased)

    def fc(self, scope_name, x, class_num, flat=False, linear=False):
        with tf.variable_scope(scope_name):
            input_shape = x.get_shape().as_list()
            if flat:
                dim = input_shape[1] * input_shape[2] * input_shape[3]
                input_processed = tf.reshape(x, [-1, dim])  # 2[batch, feature]
            else:
                dim = input_shape[1]  # already flat 2 [batch, hidden_feature]
                input_processed = x

            weights = self._get_variable(name='weights',
                                         shape=[dim, class_num],
                                         initializer=tf.contrib.layers.xavier_initializer()
                                    )
            bias = self._get_variable(name='bias',
                                      shape=[class_num],
                                      initializer=tf.constant_initializer(self.bias_fc_init))

            out = tf.add(tf.matmul(input_processed, weights), bias, name='linearout')  # [batch, class_num]

            if linear:
                return out
            else:
                return self.leaky_relu(out)

    def bn(self, x, is_training):
        return tf.layers.batch_normalization(x, training=is_training)

    def max_pool(self, x, ksize=3, stride=2):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME')

    def leaky_relu(self, x, leaky_alpha=leaky_alpha, dtype=tf.float32):
        x = tf.cast(x, dtype=dtype)
        bool_mask = (x > 0)
        mask = tf.cast(bool_mask, dtype=dtype)
        return 1.0 * mask * x + leaky_alpha * (1 - mask) * x

    def bottleneck(self, x, stride):    #stride

        input_channel=x.get_shape()[-1]
        output_channel = 4 * input_channel

        shortcut = x
        #with tf.variable_scope('a'):
        x = self.conv('a', x, ksize=1, filters_out=input_channel, stride=stride)
        #with tf.variable_scope('b'):
        x = self.conv('b', x, ksize=3, filters_out=input_channel, stride=1)
        #with tf.variable_scope('c'):
        x = self.conv('c', x, ksize=1, filters_out=output_channel, stride=1, liner=True)
        #with tf.variable_scope('shortcut'):
        shortcut = self.conv('shortcut', shortcut, ksize=1, filters_out=output_channel, stride=stride, liner=True)

        return self.leaky_relu(x + shortcut)

    def building_block(self, scope_name, x, output_channel, stride=2, reuse=None):
        with tf.variable_scope(scope_name):
            input_channel = x.get_shape()[-1]
            # output_channel = input_shape

            shortcut = x
            #with tf.variable_scope('A'):
            x = self.conv('A', x, ksize=3, filters_out=input_channel, stride=stride, reuse=reuse)
            #with tf.variable_scope('B'):
            x = self.conv('B', x, ksize=3, filters_out=output_channel, stride=1, liner=True, reuse=reuse)
            #with tf.variable_scope('Shortcut'):
            if output_channel != input_channel or stride != 1:
                shortcut = self.conv('Shortcut', shortcut, ksize=1, filters_out=output_channel, stride=stride, liner=True, reuse=reuse)

            return self.leaky_relu(x + shortcut)

    def conv_mask(self, x, mask):
        tempsize = x.get_shape().as_list()
        mask_resize = tf.image.resize_images(mask, [tempsize[1], tempsize[2]])

        return x * mask_resize

    def _normlized_0to1(self, mat): # tensor [batch_size, image_height, image_width, channels] normalize each fea map(??salency map??)
        mat_shape = mat.get_shape().as_list()
        tempmin = tf.reduce_min(mat, axis=1)
        tempmin= tf.reduce_min(tempmin, axis=1)     #each batch,each channel , the minimize of each salency map,[batch,1]  [[0.1],[0.05]...,[0.02]]
        tempmin = tf.reshape(tempmin, [-1, 1, 1, mat_shape[3]])
        tempmat = mat - tempmin     # for min=0
        tempmax = tf.reduce_max(tempmat, axis=1)
        tempmax = tf.reduce_max(tempmax, axis=1) + self.eps
        tempmax = tf.reshape(tempmax, [-1, 1, 1, mat_shape[3]])

        return tempmat / tempmax

    def _normlized(self, mat):  # tensor [batch_size, image_height, image_width, channels] normalize each fea map,  max_value to
        mat_shape = mat.get_shape().as_list()
        tempsum = tf.reduce_sum(mat, axis=1)
        tempsum = tf.reduce_sum(tempsum, axis=1) + self.eps          #each batch,each channel have a value,sum of each feature map(w*h) [batch_size, channel]
        tempsum = tf.reshape(tempsum, [-1, 1, 1, mat_shape[3]])
        return mat / tempsum


    def lstm_fc(self, scope_name, x, n_layers, n_neurons, n_outputs, sequence_length, state_is_tuple=True, state_is_zero_init=True):
        with tf.variable_scope(scope_name):
            cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, state_is_tuple=state_is_tuple) for _ in
                     range(n_layers)]  # state_is_tuple decide the size of states
            cells = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.5) for cell in cells]
            cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=state_is_tuple)

            batch_size = x.get_shape()[0]       # .as_list()
            n_steps = x.get_shape()[1]
            n_input = x.get_shape()[-1]

            if state_is_zero_init:
                if state_is_tuple:
                    lstm_state_c = tf.zeros(shape=(batch_size, n_neurons))
                    lstm_state_h = tf.zeros(shape=(batch_size, n_neurons))
                    lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=lstm_state_c, h=lstm_state_h)
                    state_init = (lstm_state,) * n_layers
                else:
                    state_init = tf.zeros([batch_size, n_layers * 2 * n_neurons])
                rnn_outputs, states = tf.nn.dynamic_rnn(cells, x, dtype=tf.float32, initial_state=state_init)
            else:
                rnn_outputs, states = tf.nn.dynamic_rnn(cells, x, sequence_length=sequence_length, dtype=tf.float32)    #sequence_length=n_steps,

            stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
            stacked_outputs_0 = tf.layers.dense(stacked_rnn_outputs, 256)
            stacked_outputs_1 = tf.layers.dense(stacked_outputs_0, 128)
            stacked_outputs_2 = tf.layers.dense(stacked_outputs_1, n_outputs)
            fc_output = tf.reshape(stacked_outputs_2, [-1, n_steps, n_outputs], name="logits")  # 预测结果

            return rnn_outputs, fc_output, states

    def convlstm_cnn_fc(self, scope_name, x, n_layers, is_training, state_is_tuple=True):
        with tf.variable_scope(scope_name+'_convlstm'):
            height=x.get_shape().as_list()[2]
            width=x.get_shape().as_list()[3]
            channel=x.get_shape().as_list()[4]
            ConvLSTMCell=tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,  # ConvLSTMCell definition
                                        input_shape=[height, width, channel],
                                        output_channels=channel,
                                        kernel_shape=[7, 7],
                                        skip_connection=False)
            cells=[ConvLSTMCell for _ in range(n_layers)]

            cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
            batch_size = x.get_shape()[0]  # .as_list()
            initial_state = cells.zero_state(batch_size, dtype=tf.float32)
            outputs, _ = tf.nn.dynamic_rnn(cells, x, initial_state=initial_state, time_major=False,dtype="float32") #[batch, n_steps, 28,28,64]

        with tf.variable_scope(scope_name+'_cnn_fc'):
            n_steps =x.get_shape().as_list()[1]
            for i in range(n_steps):
                outputs_0 = outputs[:, i, :, :, :]
                if i == 0:
                    y_0 = self.cnn('cnn', outputs_0, is_training, reuse=False)  # [batch, n_input]
                    y_0 = tf.expand_dims(y_0, 1)
                    y = y_0
                else:
                    y_0 = self.cnn('cnn', outputs_0, is_training, reuse=True)  # [batch, n_input]
                    y_0 = tf.expand_dims(y_0, 1)
                    y = tf.concat([y,y_0], 1) #[batch, n_steps, 5,5,128]
            stacked_rnn_outputs = tf.reshape(y, [-1, y.get_shape().as_list()[2]*y.get_shape().as_list()[3]*y.get_shape().as_list()[4]])
            stacked_outputs_0 = tf.layers.dense(stacked_rnn_outputs, 256)
            stacked_outputs_0 = tf.layers.dropout(stacked_outputs_0,training=is_training)
            stacked_outputs_1 = tf.layers.dense(stacked_outputs_0, 2)
            fc_output = tf.reshape(stacked_outputs_1, [-1, n_steps, 2], name="logits")  # 预测结果


        return fc_output


    def cnn(self, scope_name, x, is_training, reuse=True):
        with tf.variable_scope(scope_name):
            conv_1 = self.conv('conv_1', x, ksize=5, filters_out=128, stride=1, is_training=is_training, batch_norm=True, reuse=reuse)
            norm_1 = tf.nn.local_response_normalization(conv_1)
            pool_1 = self.max_pool(norm_1, ksize=3, stride=2)

            conv_2 = self.conv('conv_2', pool_1, ksize=3, filters_out=128, stride=1, is_training=is_training, batch_norm=True, reuse=reuse)
            norm_2 = tf.nn.local_response_normalization(conv_2)
            pool_2 = self.max_pool(norm_2, ksize=3, stride=2)


            return pool_2

    def convlstm_gap_fc(self, scope_name, x, n_layers, is_training, state_is_tuple=True):
        """

        :param scope_name:
        :param x:           B,n_steps,5,5,128
        :param n_layers: 2
        :param is_training:
        :param state_is_tuple:
        :return:
        """
        # startflagcnn = True
        # n_steps = x.get_shape().as_list()[1]
        # with tf.variable_scope(scope_name+'_convlstm_gap'):
        #     height=x.get_shape().as_list()[2]
        #     width=x.get_shape().as_list()[3]
        #     channel=x.get_shape().as_list()[4]
        #     ConvLSTMCell=tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,  # ConvLSTMCell definition
        #                                 input_shape=[height, width, channel],
        #                                 output_channels=channel,
        #                                 kernel_shape=[3, 3],
        #                                 skip_connection=False)
        #     cells=[ConvLSTMCell for _ in range(n_layers)]
        #
        #     cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
        #     batch_size = x.get_shape()[0]  # .as_list()
        #     initial_state = cells.zero_state(batch_size, dtype=tf.float32)
        #     outputs, _ = tf.nn.dynamic_rnn(cells, x, initial_state=initial_state, time_major=False,dtype="float32") #[batch, n_steps, 5,5,128]
        #     output_GAP=tf.reduce_mean(outputs,axis=(2,3)) #[batch, n_steps, 256]
        # with tf.variable_scope(scope_name + 'fc'):
        #     stacked_rnn_outputs = tf.reshape(output_GAP, [-1, output_GAP.get_shape().as_list()[2] ])
        #     stacked_outputs_0 = tf.layers.dense(stacked_rnn_outputs, 256)
        #     stacked_outputs_0 = tf.layers.dropout(stacked_outputs_0, training=is_training)
        #     stacked_outputs_1 = tf.layers.dense(stacked_outputs_0, 2)
        #     fc_output = tf.reshape(stacked_outputs_1, [-1, n_steps, 2], name="logits")  # 预测结果
        #
        # return fc_output

        startflagcnn = True
        n_steps = x.get_shape().as_list()[1]
        with tf.variable_scope(scope_name + '_convlstm_gap'):
            height = x.get_shape().as_list()[2]
            width = x.get_shape().as_list()[3]
            channel = x.get_shape().as_list()[4]
            batch_size = x.get_shape().as_list()[0]

            cell_1 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,  # ConvLSTMCell definition
                                        input_shape=[height, width, channel],
                                        output_channels=channel,
                                        kernel_shape=[3, 3],
                                        skip_connection=False)
            cell_2 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,  # ConvLSTMCell definition
                                        input_shape=[height, width, channel],
                                        output_channels=channel,
                                        kernel_shape=[3, 3],
                                        skip_connection=False)

            new_state_1 = cell_1.zero_state(batch_size, tf.float32)
            new_state_2 = cell_2.zero_state(batch_size, tf.float32)
            # print(videoslides.get_shape().as_list())
            for indexframe in range(n_steps):
                frame = x[:, indexframe, ...]
                y_1, new_state_1 = cell_1(frame, new_state_1, 'lstm_layer1')
                y_2, new_state_2 = cell_2(y_1, new_state_2,  'lstm_layer2')
                y_2_gap = tf.reduce_mean(y_2, axis=(1, 2))
                stacked_outputs_0 = tf.layers.dense(y_2_gap, 256,name='fc0')
                stacked_outputs_0 = tf.layers.dropout(stacked_outputs_0, training=is_training,name='dr0')
                stacked_outputs_1 = tf.layers.dense(stacked_outputs_0, 2,name='fc1')
                if startflagcnn == True:
                    tf.get_variable_scope().reuse_variables()
                    startflagcnn = False
                stacked_outputs_1 = tf.expand_dims(stacked_outputs_1, 1)
                if indexframe == 0:
                    fcpout = stacked_outputs_1
                else:
                    fcpout = tf.concat([fcpout, stacked_outputs_1], axis=1)

        return fcpout

    def lstm_gap_fc_timegate(self, scope_name, x, n_layers, is_training,delta_year,init_a,init_b, state_is_tuple=True):
        """

        """
        startflagcnn=True

        with tf.variable_scope(scope_name+'_lstm_gap'):
            n_steps = x.get_shape().as_list()[1]
            batch_size=x.get_shape().as_list()[0]
            channel=x.get_shape().as_list()[2]
            from tensorflow.contrib import rnn
            cell_1 = LSTMCell_timegate(num_units=channel, state_is_tuple=state_is_tuple)
            cell_2 =  LSTMCell_timegate(num_units=channel, state_is_tuple=state_is_tuple)

            new_state_1 = cell_1.zero_state(batch_size, tf.float32)
            new_state_2 = cell_2.zero_state(batch_size, tf.float32)
            # print(videoslides.get_shape().as_list())
            for indexframe in range(n_steps):
                frame = x[:, indexframe, ...]
                y_1, new_state_1,a_1,b_1 = cell_1(frame, new_state_1, delta_year[:,indexframe],init_a,init_b,'lstm_layer1')
                y_2, new_state_2,a_2,b_2 = cell_2(y_1, new_state_2, delta_year[:,indexframe],init_a,init_b,'lstm_layer2')
                # y_2_gap=tf.reduce_mean(y_2,axis=(1,2))

                y_2_gap = y_2
                stacked_outputs_0 = tf.layers.dense(y_2_gap, 256,name='fc0')
                stacked_outputs_0 = tf.layers.dropout(stacked_outputs_0, training=is_training,name='do0')
                stacked_outputs_1 = tf.layers.dense(stacked_outputs_0, 2,name='fc1')
                if  startflagcnn == True:
                    tf.get_variable_scope().reuse_variables()
                    startflagcnn = False
                stacked_outputs_1 = tf.expand_dims(stacked_outputs_1, 1)
                if indexframe == 0:
                    fcpout = stacked_outputs_1
                else:
                    fcpout = tf.concat([fcpout, stacked_outputs_1], axis=1)


        return fcpout, a_1,b_1,a_2,b_2


    def convlstm_gap_fc_timegate(self, scope_name, x, n_layers, is_training,delta_year,init_a,init_b, state_is_tuple=True):
        """
        :param scope_name:
        :param x:           B,n_steps,5,5,128
        :param n_layers: 2
        :param is_training:
        :param state_is_tuple:
        :return:
        """
        startflagcnn=True
        n_steps = x.get_shape().as_list()[1]
        with tf.variable_scope(scope_name+'_convlstm_gap'):
            height=x.get_shape().as_list()[2]
            width=x.get_shape().as_list()[3]
            channel=x.get_shape().as_list()[4]
            batch_size=x.get_shape().as_list()[0]


            cell_1 = ConvLSTMCell_timegate(conv_ndims=2,  # ConvLSTMCell definition
                                        input_shape=[height, width, channel],
                                        output_channels=channel,
                                        kernel_shape=[3, 3],
                                        skip_connection=False)
            cell_2 = ConvLSTMCell_timegate(conv_ndims=2,  # ConvLSTMCell definition
                                        input_shape=[height, width, channel],
                                        output_channels=channel,
                                        kernel_shape=[3, 3],
                                        skip_connection=False)

            new_state_1 = cell_1.zero_state(batch_size, 2, tf.float32)
            new_state_2 = cell_2.zero_state(batch_size, 2, tf.float32)
            # print(videoslides.get_shape().as_list())
            for indexframe in range(n_steps):
                frame = x[:, indexframe, ...]
                y_1, new_state_1 = cell_1(frame, new_state_1, delta_year[:,indexframe],init_a,init_b, 'lstm_layer1')
                y_2, new_state_2 = cell_2(y_1, new_state_2, delta_year[:,indexframe],init_a,init_b,'lstm_layer2')
                # y_2_gap=tf.reduce_mean(y_2,axis=(1,2))
                y_2_shape = y_2.get_shape().as_list()
                y_2_gap = tf.reshape(y_2, shape=[-1, y_2_shape[1] * y_2_shape[2] * y_2_shape[3]])
                stacked_outputs_0 = tf.layers.dense(y_2_gap, 256,name='fc0')
                stacked_outputs_0 = tf.layers.dropout(stacked_outputs_0, training=is_training,name='do0')
                stacked_outputs_1 = tf.layers.dense(stacked_outputs_0, 2,name='fc1')
                if  startflagcnn == True:
                    tf.get_variable_scope().reuse_variables()
                    startflagcnn = False
                stacked_outputs_1 = tf.expand_dims(stacked_outputs_1, 1)
                if indexframe == 0:
                    fcpout = stacked_outputs_1
                else:
                    fcpout = tf.concat([fcpout, stacked_outputs_1], axis=1)


        return fcpout



    """calculate_acc_sensitivity_specificity in one batch"""
    def _result(self, labGTs, predictions):     #calculate on each validation batch       #label_predict_op=(batch, n_steps, n_outputs)
        e = math.e
        #print(labGTs)
        #print(predictions)
        batch_size_1 = labGTs.shape[0]      #ndarray
        t = 0

        n_steps1 = labGTs.shape[1]
        n_steps2 = predictions.shape[1]
        assert n_steps1 == n_steps2
        for i in range(batch_size_1):
            for j in range(n_steps1):
                prediction_0 = e ** (predictions[i, j, 0]) / (e ** (predictions[i, j, 1]) + e ** (predictions[i, j, 0]) )
                prediction_1 = e ** (predictions[i, j, 1]) / (e ** (predictions[i, j, 1]) + e ** (predictions[i, j, 0]) )

                assert labGTs[i, j] == 1 or labGTs[i, j] == 0# or labGTs[i, j] == 2
                if labGTs[i, j] == 1:
                    if prediction_1 >= prediction_0: #and prediction_1 >= prediction_2:
                        t = t + 1
                elif labGTs[i, j] == 0:
                    if prediction_0 >= prediction_1: #and prediction_0 >= prediction_2:
                        t = t + 1

        acc = t / (batch_size_1 * n_steps1)
        return acc



