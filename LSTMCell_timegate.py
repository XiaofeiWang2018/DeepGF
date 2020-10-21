from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import numbers

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

from tensorflow.contrib.rnn import *
import tensorflow as tf
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
weight_decay = 5*1e-6
import numpy as np
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
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

class _Linear(object):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of weight variable.
    dtype: data type for variables.
    build_bias: boolean, whether to build a bias variable.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.

  Raises:
    ValueError: if inputs_shape is wrong.
  """

  def __init__(self,
               args,
               output_size,
               build_bias,
               bias_initializer=None,
               kernel_initializer=None):
    self._build_bias = build_bias

    if args is None or (nest.is_sequence(args) and not args):
      raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
      args = [args]
      self._is_sequence = False
    else:
      self._is_sequence = True

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
      if shape.ndims != 2:
        raise ValueError("linear is expecting 2D arguments: %s" % shapes)
      if shape[1].value is None:
        raise ValueError("linear expects shape[1] to be provided for shape %s, "
                         "but saw %s" % (shape, shape[1]))
      else:
        total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
      self._weights = vs.get_variable(
          _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
          dtype=dtype,
          initializer=kernel_initializer)
      if build_bias:
        with vs.variable_scope(outer_scope) as inner_scope:
          inner_scope.set_partitioner(None)
          if bias_initializer is None:
            bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
          self._biases = vs.get_variable(
              _BIAS_VARIABLE_NAME, [output_size],
              dtype=dtype,
              initializer=bias_initializer)

  def __call__(self, args):
    if not self._is_sequence:
      args = [args]

    if len(args) == 1:
      res = math_ops.matmul(args[0], self._weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), self._weights)
    if self._build_bias:
      res = nn_ops.bias_add(res, self._biases)
    return res



class LSTMCell_timegate(BasicLSTMCell):

    def __init__(self,num_units,state_is_tuple):
        super(LSTMCell_timegate, self).__init__(num_units=num_units,state_is_tuple=state_is_tuple)

    def __call__(self, inputs, state,delta_year,init_a,init_b, scope=None):
        """Long short-term memory cell (LSTM).

        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size x 2 * self.state_size]`.

        Returns:
          A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        sigmoid = math_ops.sigmoid
        batch_size=inputs.get_shape().as_list()[0]
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

        if self._linear is None:
            with tf.variable_scope(scope+'liner'):
                self._linear = _Linear([inputs, h], 4 * self._num_units, True)
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(
            value=self._linear([inputs, h]), num_or_size_splits=4, axis=1)
        """time gate"""
        with tf.variable_scope(scope + '_w'):
            a = _get_variable('time_gate_a', shape=[1], initializer=tf.constant_initializer(init_a))
            b = _get_variable('time_gate_b', shape=[1], initializer=tf.constant_initializer(init_b))
        w = tf.sigmoid(delta_year * a + b)
        w = tf.reshape(w, [batch_size, 1])
        """time gate"""
        new_c = (c * sigmoid(f*w + self._forget_bias) + sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * sigmoid(o)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state,a,b

class LSTMCell_timegateKDD(BasicLSTMCell):

    def __init__(self,num_units,state_is_tuple):
        super(LSTMCell_timegateKDD, self).__init__(num_units=num_units,state_is_tuple=state_is_tuple)

    def __call__(self, inputs, state,delta_year,init_a,init_b, scope=None):
        """Long short-term memory cell (LSTM).

        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size x 2 * self.state_size]`.

        Returns:
          A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        sigmoid = math_ops.sigmoid
        batch_size=inputs.get_shape().as_list()[0]
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

        if self._linear is None:
            with tf.variable_scope(scope+'liner'):
                self._linear = _Linear([inputs, h], 4 * self._num_units, True)
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(
            value=self._linear([inputs, h]), num_or_size_splits=4, axis=1)


        """time gate"""
        with tf.variable_scope(scope + '_w'):

            w = _get_variable('time_gate_w', shape=[320,320], initializer=tf.random_normal_initializer(mean=0.0,stddev=1.0,dtype=tf.float32))
            b = _get_variable('time_gate_b', shape=[batch_size,320], initializer=tf.random_normal_initializer(mean=0.0,stddev=1.0,dtype=tf.float32))
        c_KDD = self._activation(tf.matmul(c,w) + b)

        """time gate"""
        delta_year=tf.tile(tf.reshape(delta_year,[batch_size,1]),[1,320])
        aa=(1/tf.log(delta_year+0.1) -1)*c_KDD
        new_c = ((c+aa) * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * sigmoid(o)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state

