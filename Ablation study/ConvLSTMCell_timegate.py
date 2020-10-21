from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.contrib.compiler import jit
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.contrib.rnn import *
import tensorflow as tf

weight_decay = 5*1e-6
def _get_variable(
                  name,
                  shape,
                  initializer,
                  weight_decay=weight_decay,
                  dtype='float32',
                  trainable=True, AAAI_VARIABLES=None):  # pretrain/ initial/

    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None

    collection = [tf.GraphKeys.GLOBAL_VARIABLES]  # , LL_VARIABLES

    return tf.get_variable(name=name,
                           shape=shape,
                           initializer=initializer,
                           regularizer=regularizer,
                           collections=collection,
                           dtype=dtype,
                           trainable=trainable,
                           )
class ConvLSTMCell_timegate(ConvLSTMCell):

    def __init__(self,conv_ndims,input_shape,output_channels,kernel_shape,skip_connection):
        super(ConvLSTMCell_timegate, self).__init__(conv_ndims,input_shape,output_channels,kernel_shape,skip_connection)

    def zero_state(self, batch_size, hiddennum, dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          tensor of shape '[batch_size x shape[0] x shape[1] x num_features]
          filled with zeros
        """

        shape = self._input_shape
        num_features = self._output_channels

        zeros = tf.zeros([batch_size, shape[0], shape[1], num_features * hiddennum])
        return zeros

    def __call__(self, inputs, state,delta_year,init_a,init_b, scope=None):
        cell, hidden = tf.split(axis=3, num_or_size_splits=2, value=state)
        new_hidden = _conv([inputs, hidden],
                           self._kernel_shape,
                           4 * self._output_channels,
                           self._use_bias,
                           scope)
        gates = array_ops.split(value=new_hidden,
                                num_or_size_splits=4,
                                axis=self._conv_ndims + 1)

        input_gate, new_input, forget_gate, output_gate = gates
        """time gate"""
        with tf.variable_scope(scope+'_w'):
            a = _get_variable('time_gate_a', shape=[1], initializer=tf.constant_initializer(init_a))
            b=_get_variable('time_gate_b',shape=[1],initializer=tf.constant_initializer(init_b))
        w=tf.sigmoid(delta_year*a+b)
        w=tf.reshape(w,[4,1,1,1])
        forget_gate=w*forget_gate
        """time gate"""
        new_cell = math_ops.sigmoid(forget_gate + self._forget_bias) * cell
        new_cell += math_ops.sigmoid(input_gate) * math_ops.tanh(new_input)
        output = math_ops.tanh(new_cell) * math_ops.sigmoid(output_gate)

        if self._skip_connection:
            output = array_ops.concat([output, inputs], axis=-1)
        new_state = tf.concat([new_cell, output],axis=3)
        return output, new_state



def _conv(args,
          filter_size,
          num_features,
          bias,
            scope,
          bias_start=0.0):
  """convolution:
  Args:
    args: a Tensor or a list of Tensors of dimension 3D, 4D or 5D,
    batch x n, Tensors.
    filter_size: int tuple of filter height and width.
    num_features: int, number of features.
    bias_start: starting value to initialize the bias; 0 by default.
  Returns:
    A 3D, 4D, or 5D Tensor with shape [batch ... num_features]
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """

  # Calculate the total size of arguments on dimension 1.
  total_arg_size_depth = 0
  shapes = [a.get_shape().as_list() for a in args]
  shape_length = len(shapes[0])
  for shape in shapes:
    if len(shape) not in [3,4,5]:
      raise ValueError("Conv Linear expects 3D, 4D or 5D arguments: %s" % str(shapes))
    if len(shape) != len(shapes[0]):
      raise ValueError("Conv Linear expects all args to be of same Dimension: %s" % str(shapes))
    else:
      total_arg_size_depth += shape[-1]
  dtype = [a.dtype for a in args][0]

  # determine correct conv operation
  if   shape_length == 3:
    conv_op = nn_ops.conv1d
    strides = 1
  elif shape_length == 4:
    conv_op = nn_ops.conv2d
    strides = shape_length*[1]
  elif shape_length == 5:
    conv_op = nn_ops.conv3d
    strides = shape_length*[1]
  with tf.variable_scope(scope):
      # Now the computation.
      kernel = vs.get_variable(
          "kernel",
          filter_size + [total_arg_size_depth, num_features],
          dtype=dtype)
      if len(args) == 1:
        res = conv_op(args[0],
                      kernel,
                      strides,
                      padding='SAME')
      else:
        res = conv_op(array_ops.concat(axis=shape_length-1, values=args),
                      kernel,
                      strides,
                      padding='SAME')
      if not bias:
        return res
      bias_term = vs.get_variable(
          "biases", [num_features],
          dtype=dtype,
          initializer=init_ops.constant_initializer(
              bias_start, dtype=dtype))
  return res + bias_term
