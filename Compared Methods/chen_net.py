
import BasicNet
import tensorflow as tf

class ChenNet(BasicNet.BasicNet):
    init_learning_rate = 10 ** (-6)

    def __init__(self):
        super(ChenNet, self).__init__()

        # self.global_step = tf.Variable(0, trainable= False)
        # self.initial_var_collection.append(self.global_step)   #???
        self.predict_label = []

        self.loss = []
        self.loss_weight = []
        self.train = []

    def inference(self, x,reuse):
        conv_1 = self.conv('conv_1', x, ksize=11, filters_out=96, stride=4, batch_norm = False,reuse=reuse)
        norm_1 = tf.nn.local_response_normalization(conv_1)
        pool_1 = self.max_pool(norm_1 , ksize = 3, stride=2)

        conv_2 = self.conv('conv_2', pool_1, ksize=5, filters_out=192, stride=1, batch_norm = False,reuse=reuse)
        norm_2 = tf.nn.local_response_normalization(conv_2)
        pool_2 = self.max_pool(norm_2, ksize=3, stride=2)

        conv_3 = self.conv('conv_3', pool_2, ksize=3, filters_out=192, stride=1, batch_norm=False,reuse=reuse)
        pool_3 = self.max_pool(conv_3, ksize=3, stride=2)

        conv_4 = self.conv('conv_4', pool_3, ksize=3, filters_out=192, stride=1, batch_norm=False,reuse=reuse)
        pool_4 = self.max_pool(conv_4, ksize=3, stride=2)
        pool_4=tf.reshape(pool_4,shape=(-1,pool_4.get_shape().as_list()[1]*pool_4.get_shape().as_list()[2]*pool_4.get_shape().as_list()[3]))
        stacked_outputs_0 = tf.layers.dense(pool_4, 256, name='fc0', reuse=reuse)
        stacked_outputs_0 = tf.layers.dropout(stacked_outputs_0, training=True, name='do0')
        fc1 = tf.layers.dense(stacked_outputs_0, 2, name='fc1', reuse=reuse)

        self.predict_label = fc1

        return fc1,pool_4


    def _loss(self, label):
        batch_size = label.get_shape().as_list()[0]
        weight_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=None)
        loss_weight = tf.add_n(weight_loss)

        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict_label, labels=label)        #labels is in form of 1 dim,[batch_size], logistics is in 2 dims, [batch_size, output_num]
        # print(ce.get_shape().as_list())
        loss_label = tf.reduce_sum(ce)

        self.loss_label = loss_label / batch_size
        self.loss_weight = loss_weight

        loss = self.loss_weight + self.loss_label
        self.loss = loss

        tf.summary.scalar('loss', self.loss)

        return self.loss, self.loss_label, self.loss_weight



    def _loss_val(self, label):
        batch_size = label.get_shape().as_list()[0]
        weight_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=None)
        loss_weight = tf.add_n(weight_loss)

        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict_label,
                                                            labels=label)  # labels is in form of 1 dim,[batch_size], logistics is in 2 dims, [batch_size, output_num]
        # print(ce.get_shape().as_list())
        loss_label = tf.reduce_sum(ce)

        self.loss_label = loss_label / batch_size
        self.loss_weight = loss_weight

        loss = self.loss_weight + self.loss_label
        self.loss = loss

        return self.loss

    def _train(self):

        opt = tf.train.AdamOptimizer(self.init_learning_rate, beta1=0.9, beta2=0.999, epsilon= 1e-08)

        gradients = opt.compute_gradients(self.loss)    #all variables are trainable
        apply_gradient_op = opt.apply_gradients(gradients)      #, global_step=self.global_step
        self.train = apply_gradient_op

        return apply_gradient_op

    def _top_k_accuracy(self, labels, k=1):
        print('inacc',self.predict_label)
        batch_size = labels.get_shape().as_list()[0]
        right_1 = tf.to_float(tf.nn.in_top_k(predictions=self.predict_label, targets=labels, k=k))
        num_correct_per_batch = tf.reduce_sum(right_1)
        acc = num_correct_per_batch / batch_size

        tf.summary.scalar('acc', acc)

        return acc