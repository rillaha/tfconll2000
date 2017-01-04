import tensorflow as tf
import numpy as np

def BidirectionalEncoder(sequence_input, sequence_length, lstm_single_dim, encode_dim):
    forward_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_single_dim, forget_bias=0.0)
    backward_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_single_dim, forget_bias=0.0)
    forward_weight = tf.Variable(tf.random_uniform([lstm_single_dim, encode_dim], -1.0, 1.0))
    backward_weight = tf.Variable(tf.random_uniform([lstm_single_dim, encode_dim], -1.0, 1.0))
    bias = tf.Variable(tf.random_uniform([encode_dim], -1.0, 1.0))
    both_sequence_output, both_final_state = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, sequence_input, dtype=tf.float32, sequence_length=sequence_length)
    forward_final_state = both_final_state[0]
    forward_output = forward_final_state[1]
    backward_output = both_sequence_output[1][:,0,:]
    encoded = tf.matmul(forward_output, forward_weight) + tf.matmul(backward_output, backward_weight) + bias
    parameters = [forward_cell, backward_cell, forward_weight, backward_weight, bias]
    return encoded, parameters

def BidirectionalLSTMLayer(sequence_input, sequence_length, lstm_single_dim, output_dim):
    forward_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_single_dim, forget_bias=0.0)
    backward_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_single_dim, forget_bias=0.0)
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
    return layer_output, parameters
