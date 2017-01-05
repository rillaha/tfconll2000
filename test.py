import time
import tensorflow as tf
import numpy as np
import utils

# import conllreader as cr

# indexer = cr.Indexer()
# print("read train")
# train_data = cr.ConllDataset("corpus/train.head.txt", indexer)
# print("read dev")
# dev_data = cr.ConllDataset("corpus/train.tail.txt", indexer)
# print("read test")
# test_data = cr.ConllDataset("corpus/test.txt", indexer)
# print("update")
# indexer.update()
# print("len char:%d  len POS:%d  len chunk:%d" % (len(indexer.dic["char"]), len(indexer.dic["POS"]), len(indexer.dic["chunk"])))

# incremental and dynamic_rnn vs rnn
"""

batch_size = 3
max_seq_length = 100
input_dim = 100
inp = Rand([batch_size, max_seq_length, input_dim])
#cell = utils.IncrementalLSTMCell(num_units=700, dim_incremental=500)
#cell2 = utils.IncrementalLSTMCell(num_units=700, dim_incremental=500)
with tf.variable_scope("RNN1"):
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=700)
with tf.variable_scope("RNN2"):
    cell2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=700)
l = tf.placeholder(tf.int32, [batch_size])
outs, state, state2 = tf.nn.bidirectional_rnn(cell, cell2, [inp[:,i,:] for i in range(max_seq_length)], dtype=tf.float64, sequence_length=l)
#outs, state = tf.nn.dynamic_rnn(cell,inp, dtype=tf.float64, sequence_length=l)
init()
#fd = {l:np.array([max_seq_length for i in range(batch_size)])}
fd = {l:np.array([3 for i in range(batch_size)])}
#fd = {l:np.array([0 for i in range(batch_size-1)] + [max_seq_length])}
print(outs)
t = time.time()
print(t)
for i in range(10):
    t = time.time()
    sess.run([outs, state], feed_dict=fd)
    ot = time.time()
    #print(o)
    #print(s)
    print(ot -t)

a,b,c = sess.run([outs, state, state2], feed_dict=fd)
u = a[2]
v = b.h
w = c.h
"""

def Rand(arg):
    return tf.constant(np.random.random(arg))

vec = Rand([3])
mat = Rand([2,3])

sess = tf.InteractiveSession()
def init():
    return sess.run(tf.initialize_all_variables())
run = sess.run

