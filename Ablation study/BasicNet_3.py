
import tensorflow as tf
import numpy as np
from PIL import Image
# #from config import Config




class BasicNet(object):
    weight_decay = 5*1e-6
    bias_conv_init = 0.1 #weight init for biasis
    bias_fc_init = 0.1
    leaky_alpha = 0.1
    is_training = True




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

    def conv(self, scope_name, x, ksize, filters_out, stride=1, batch_norm= True, liner = False, reuse=None):
        with tf.variable_scope(scope_name, reuse=reuse):
            filters_in = x.get_shape()[-1].value

            shape = [ksize, ksize, filters_in, filters_out]  # conv kernel size
            weights = self._get_variable('weights',
                                    shape=shape,
                                    initializer=tf.contrib.layers.xavier_initializer()  # need to set seed number
                                    )
            tf.add_to_collection('conv_weight',weights)
            bias = self._get_variable('bias',
                                 shape=[filters_out],
                                 initializer=tf.constant_initializer(self.bias_conv_init)
                                 )
            tf.add_to_collection('conv_bias', bias)
            conv = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
            conv_bias = tf.nn.bias_add(conv, bias, name='linearout')

            if batch_norm:
                out = self.bn(conv_bias, is_training=self.is_training)
            else:
                out = conv_bias

            if liner:
                return out
            else:
                return self.leaky_relu(out)

    def spe_conv(self, scope_name, x, ksize1, ksize2, filters_out, stride=1, batch_norm= True, liner = False, reuse=None):
        with tf.variable_scope(scope_name, reuse=reuse):
            filters_in = x.get_shape()[-1].value

            shape = [ksize1, ksize2, filters_in, filters_out]  # conv kernel size
            weights = self._get_variable('weights',
                                    shape=shape,
                                    initializer=tf.contrib.layers.xavier_initializer()  # need to set seed number
                                    )
            tf.add_to_collection('conv_weight', weights)
            bias = self._get_variable('bias',
                                 shape=[filters_out],
                                 initializer=tf.constant_initializer(self.bias_conv_init)
                                 )
            tf.add_to_collection('conv_bias', bias)
            conv = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
            conv_bias = tf.nn.bias_add(conv, bias, name='linearout')

            if batch_norm:
                out = self.bn(conv_bias, is_training=self.is_training)
            else:
                out = conv_bias

            if liner:
                return out
            else:
                return self.leaky_relu(out)

    def dconv(self, scope_name, x, ksize, filters_out, stride=1, liner=False, reuse=None):
        with tf.variable_scope(scope_name, reuse=reuse):
            filters_in = x.get_shape()[-1].value

            shape = [ksize, ksize, filters_out, filters_in]
            weights = self._get_variable('weights',
                                    shape=shape,
                                    initializer=tf.contrib.layers.xavier_initializer()
                                    )
            tf.add_to_collection('dconv_weight', weights)
            bias = self._get_variable('bias',
                                 shape=[filters_out],
                                 initializer=tf.constant_initializer(self.bias_conv_init)
                                 )
            tf.add_to_collection('dconv_bias', bias)

            output_shape = tf.stack([tf.shape(x)[0], tf.shape(x)[1] * stride, tf.shape(x)[2] * stride, filters_out])
            conv = tf.nn.conv2d_transpose(x, weights, output_shape, strides=[1, stride, stride, 1], padding='SAME')
            conv_biased = tf.nn.bias_add(conv, bias, name='linearout')

            if liner:
                return conv_biased
            else:
                return self.leaky_relu(conv_biased)

    def fc(self, scope_name, x, class_num, flat=False, linear=False, reuse=None):
        with tf.variable_scope(scope_name, reuse=reuse):
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
            tf.add_to_collection('fc_weight', weights)
            bias = self._get_variable(name='bias',
                                      shape=[class_num],
                                      initializer=tf.constant_initializer(self.bias_fc_init))
            tf.add_to_collection('fc_bias', bias)
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
        return tf.nn.relu(x)    #1.0 * mask * x + leaky_alpha * (1 - mask) * x

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
            x = self.conv('A', x, ksize=3, filters_out=output_channel, stride=stride, reuse=reuse)
            #with tf.variable_scope('B'):
            x = self.conv('B', x, ksize=3, filters_out=output_channel, stride=1, liner=True, reuse=reuse)
            #with tf.variable_scope('Shortcut'):
            if output_channel != input_channel or stride != 1:
                shortcut = self.conv('Shortcut', shortcut, ksize=1, filters_out=output_channel, stride=stride, liner=True, reuse=reuse)

            return self.leaky_relu(x + shortcut)

    def v_building_block(self, scope_name, x, output_channel, stride=2, reuse=None):
        with tf.variable_scope(scope_name):
            input_channel = x.get_shape()[-1]
            # output_channel = input_shape

            shortcut = x
            x1 = self.conv('A', x, ksize=3, filters_out=input_channel, stride=stride, reuse=reuse)
            x1 = self.conv('B', x1, ksize=3, filters_out=input_channel, stride=1, liner=True, reuse=reuse)

            x2 = self.conv('C', x, ksize=3, filters_out=input_channel, stride=stride, liner=True, reuse=reuse )

            _x = tf.concat([x1, x2], 3)

            x = self.conv('D' ,_x, ksize=1, filters_out=output_channel, stride=1, liner=True, reuse=reuse)

            if output_channel != input_channel or stride != 1:
                shortcut = self.conv('Shortcut', shortcut, ksize=1, filters_out=output_channel, stride=stride, liner=True, reuse=reuse)

            return self.leaky_relu(x + shortcut)


    def vv_building_block(self, scope_name, x, output_channel, stride=2, reuse=None):
        with tf.variable_scope(scope_name):
            input_channel = x.get_shape()[-1]
            # output_channel = input_shape

            shortcut = x
            x1 = self.conv('A', x, ksize=3, filters_out=input_channel, stride=stride, reuse=reuse)
            x1 = self.conv('B', x1, ksize=3, filters_out=input_channel, stride=1, liner=True, reuse=reuse)

            x2 = self.conv('C', x, ksize=3, filters_out=input_channel, stride=stride, liner=True, reuse=reuse)

            #x3 = self.conv('D', x, ksize=5, filters_out=output_channel, stride=stride, liner=True, reuse=reuse)

            x3 = self.spe_conv('D1', x, ksize1=5, ksize2=1, filters_out=input_channel, stride=1, liner=True, reuse=reuse)
            #print('x31',x3)
            x3 = self.spe_conv('D2', x3, ksize1=1, ksize2=5, filters_out=input_channel, stride=stride, liner=True,
                               reuse=reuse)
            #print('x32',x3)
            x4 = self.spe_conv('E1', x, ksize1=7, ksize2=1, filters_out=input_channel, stride=1, liner=True, reuse=reuse)
            x4 = self.spe_conv('E2', x4, ksize1=1, ksize2=7, filters_out=input_channel, stride=stride, liner=True, reuse=reuse)

            _x = tf.concat([x1, x2, x3, x4], 3)

            x = self.conv('F' ,_x, ksize=1, filters_out=output_channel, stride=1, liner=True, reuse=reuse)

            if output_channel != input_channel or stride != 1:
                shortcut = self.conv('Shortcut', shortcut, ksize=1, filters_out=output_channel, stride=stride, liner=True, reuse=reuse)

            return self.leaky_relu(x + shortcut)

    def conv_mask(self, x, mask):
        tempsize = x.get_shape().as_list()
        mask_resize = tf.image.resize_images(mask, [tempsize[1], tempsize[2]])

        return x * mask_resize

    def _normlized_0to1(self, mat): # tensor [batch_size, image_height, image_width, channels] normalize each fea map(??salency map??)
        mat_shape = mat.get_shape().as_list()
        #print('mat_shape    ',mat_shape)
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

    def wgn(self, x, snr):
        snr = 10 ** (snr / 10)
        x_shape = x.get_shape().as_list()
        length = x_shape[0]*x_shape[1]*x_shape[2]*x_shape[3]
        xpower = x**2
        xpower = tf.reduce_sum(xpower, axis=0)
        xpower = tf.reduce_sum(xpower, axis=0)
        xpower = tf.reduce_sum(xpower, axis=0)
        xpower = tf.reduce_sum(xpower, axis=0)
        #print(xpower.shape)
        npower = xpower / snr
        noise = np.random.randn(length) * tf.sqrt(tf.cast(npower, tf.float32))
        noise = tf.reshape(noise,[x_shape[0], x_shape[1], x_shape[2], x_shape[3]])

        return x + noise

    def crop(self, x, percent=0.75):# percent 为保留的比例，如percent=0.7为剪裁后保留原图的70%
        x_shape = x.get_shape().as_list()  # 四维
        ori_h = x_shape[1]
        ori_w = x_shape[2]
        batch = x_shape[0]
        print(ori_h)
        #print(type(ori_h) == 'NoneType')
        if ori_h == None:
            ori_h = 112
            ori_w = 112
            batch = 8
        #print(ori_h)
        h_diet = int(float(ori_h) * percent)
        w_diet = int(float(ori_w) * percent)
        # h_start = int(float(ori_h) * (1-percent)/2)
        # h_end = int(float(ori_h) * (1 + percent)/2)
        # w_start = int(float(ori_w) * (1-percent)/2)
        # w_end = int(float(ori_w) * (1 + percent)/2)
        for n in range(batch):
            # x1 = x[n, h_start:h_end, w_start:w_end, :]
            # x_uint8 = tf.cast(x1*255, tf.uint8)
            # x_uint8_ori_size = tf.image.resize_images(x_uint8, [ori_h, ori_w])
            # x_float32_ori_size = tf.cast(x_uint8_ori_size, tf.float32)
            # x_float32_01_ori_size = x_float32_ori_size/255
            x_float32_01_ori_size = tf.image.resize_image_with_crop_or_pad(x[n, :,:,:], h_diet, w_diet)
            if n==0:
                a = x_float32_01_ori_size
            elif n==1:
                b = x_float32_01_ori_size
            elif n==2:
                c = x_float32_01_ori_size
            elif n==3:
                d = x_float32_01_ori_size
            elif n==4:
                e = x_float32_01_ori_size
            elif n == 5:
                f = x_float32_01_ori_size
            elif n == 6:
                g = x_float32_01_ori_size
            elif n == 7:
                h = x_float32_01_ori_size

        print(tf.stack([a,b,c,d,e,f,g,h], axis=0))
        return tf.stack([a,b,c,d,e,f,g,h], axis=0)

    def symmetry(self, x):#左右翻转 [batch_size, image_height, image_width, channels]
        x_sym = x[:, :, ::-1, :]
        #print('x_sym',x_sym)
        return x_sym