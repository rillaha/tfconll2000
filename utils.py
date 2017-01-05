import collections
import tensorflow as tf
import numpy as np

def BidirectionalEncoder(sequence_input, sequence_length, max_sequence_length, lstm_single_dim, encode_dim):
    forward_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_single_dim, forget_bias=0.0)
    backward_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_single_dim, forget_bias=0.0)
    forward_weight = tf.Variable(tf.random_uniform([lstm_single_dim, encode_dim], -1.0, 1.0))
    backward_weight = tf.Variable(tf.random_uniform([lstm_single_dim, encode_dim], -1.0, 1.0))
    bias = tf.Variable(tf.random_uniform([encode_dim], -1.0, 1.0))

    # tf.nn.bidirectional_dynamic_rnn version
    """
    both_sequence_output, both_final_state = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, sequence_input, dtype=tf.float32, sequence_length=sequence_length)
    forward_final_state, backward_final_state = both_final_state
    """

    # tf.nn.bidirectional_rnn version
    both_sequence_output, forward_final_state, backward_final_state = tf.nn.bidirectional_rnn(forward_cell, backward_cell, [sequence_input[:,step,:] for step in range(max_sequence_length)], dtype=tf.float32, sequence_length=sequence_length)

    forward_output = forward_final_state.h
    backward_output = backward_final_state.h
    encoded = tf.matmul(forward_output, forward_weight) + tf.matmul(backward_output, backward_weight) + bias
    parameters = [forward_cell, backward_cell, forward_weight, backward_weight, bias]
    return encoded, parameters

def BidirectionalLSTMLayer(sequence_input, sequence_length, max_sequence_length, lstm_single_dim, output_dim):
    forward_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_single_dim, forget_bias=0.0)
    backward_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_single_dim, forget_bias=0.0)

    # tf.nn.bidirectional_dynamic_rnn version
    """
    forward_weight = tf.Variable(tf.random_uniform([lstm_single_dim, output_dim], -1.0, 1.0))
    backward_weight = tf.Variable(tf.random_uniform([lstm_single_dim, output_dim], -1.0, 1.0))
    bias = tf.Variable(tf.random_uniform([output_dim], -1.0, 1.0))
    both_sequence_output, both_final_state = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, sequence_input, dtype=tf.float32, sequence_length=sequence_length)
    forward_output_flat = tf.reshape(both_sequence_output[0], [-1, lstm_single_dim])
    backward_output_flat = tf.reshape(both_sequence_output[1], [-1, lstm_single_dim])
    layer_output_flat = tf.matmul(forward_output_flat, forward_weight) + tf.matmul(backward_output_flat, backward_weight) + bias
    batch_size_op = tf.shape(sequence_input)[0]
    layer_output = tf.reshape(layer_output_flat, [batch_size_op, -1, output_dim])
    parameters = [forward_cell, backward_cell, forward_weight, backward_weight, bias]
    """

    # tf.nn.bidirectional_rnn version
    weight = tf.Variable(tf.random_uniform([lstm_single_dim * 2, output_dim], -1.0, 1.0))
    bias = tf.Variable(tf.random_uniform([output_dim], -1.0, 1.0))
    both_sequence_output, forward_final_state, backward_final_state = tf.nn.bidirectional_rnn(forward_cell, backward_cell, [sequence_input[:,step,:] for step in range(max_sequence_length)], dtype=tf.float32, sequence_length=sequence_length)
    both_sequence_output_flat = tf.reshape(tf.pack(both_sequence_output, axis=1), [-1, lstm_single_dim*2])
    layer_output_flat = tf.matmul(both_sequence_output_flat, weight) + bias
    layer_output = tf.reshape(layer_output_flat, [-1, max_sequence_length, output_dim])
    parameters = [forward_cell, backward_cell, weight, bias]

    return layer_output, parameters

_IncrementalLSTMStateTuple = collections.namedtuple("IncrementalLSTMStateTuple", ("c", "h", "incremental"))
class IncrementalLSTMStateTuple(_IncrementalLSTMStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        (c, h, i) = self
        if not c.dtype == h.dtype == i.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s vs %s" %
                            (str(c.dtype), str(h.dtype), str(i.dtype)))
        return c.dtype

class IncrementalLSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, dim_incremental, forget_bias=1.0, activation=tf.nn.tanh):
        self._num_units = num_units
        self._dim_incremental = dim_incremental
        self._forget_bias = forget_bias
        self._state_is_tuple = True
        self._activation = activation

    @property
    def state_size(self):
        return IncrementalLSTMStateTuple(self._num_units, self._num_units, self._dim_incremental)

    @property
    def output_size(self):
        return self._dim_incremental

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h, incremental = state

            with tf.variable_scope("LSTM"):
                concat = tf.nn.rnn_cell._linear([inputs, h, incremental], 4 * self._num_units, True)
                i, j, f, o = tf.nn.array_ops.split(1, 4, concat)
                new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) * self._activation(j))
                new_h = self._activation(new_c) * tf.nn.sigmoid(o)

            with tf.variable_scope("incremental"):
                u = tf.nn.rnn_cell._linear([new_h], self._dim_incremental, True)
                y = self._activation(u)

            new_state = IncrementalLSTMStateTuple(new_c, new_h, y)
            return y, new_state

#def Decoder

