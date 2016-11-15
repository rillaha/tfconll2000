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

import tensorflow as tf
import numpy as np

def Rand(arg):
    return tf.constant(np.random.random(arg))

vec = Rand([3])
mat = Rand([2,3])

sess = tf.InteractiveSession()
def init():
    return sess.run(tf.initialize_all_variables())
run = sess.run

