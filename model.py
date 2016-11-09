#!/usr/bin/python3
import sys
import tensorflow as tf
import numpy as np
import conllreader

usePWLoss = False

np.random.seed(0)
tf.set_random_seed(0)

batch_size = 2
max_sentence_length = 78
max_word_length = 60
char_size = 50
char_vocab = 256
c2w_single_size = 100
word_size = 150
encode_single_size = 100
encode_size = 100
lm_size = 100
num_label = 24
label_repr_size = encode_size
beam_width = 10

batch_size = 4
max_sentence_length = 78
max_word_length = 60
char_size = 2
char_vocab = 256
c2w_single_size = 3
word_size = 5
encode_single_size = 5
encode_size = 2
lm_size = 3
num_label = 24
label_repr_size = encode_size
beam_width = 10

print("read dataset", end="")
sys.stdout.flush()
indexer = conllreader.Indexer()
print(".", end="")
sys.stdout.flush()
train_data = conllreader.ConllDataset("corpus/train.head.txt", indexer)
print(".", end="")
sys.stdout.flush()
dev_data = conllreader.ConllDataset("corpus/train.tail.txt", indexer)
print(".", end="")
sys.stdout.flush()
test_data = conllreader.ConllDataset("corpus/test.txt", indexer)
print(".", end="")
sys.stdout.flush()
indexer.update()
print("finish")


print("make graph")

is_train_op = tf.placeholder(tf.bool)

x = tf.placeholder(tf.int32, [batch_size, max_sentence_length, max_word_length])
l_char = tf.placeholder(tf.int32, [batch_size, max_sentence_length])
l_word = tf.placeholder(tf.int32, [batch_size])
gold = tf.placeholder(tf.int32, [batch_size, max_sentence_length])

W_char_emb = tf.Variable(tf.random_uniform([char_vocab, char_size], -1.0, 1.0))
char_vec = tf.nn.embedding_lookup(W_char_emb, x)

# c2w
with tf.variable_scope("c2w"):
    c2w_fcell = tf.nn.rnn_cell.BasicLSTMCell(c2w_single_size, forget_bias=0.0)
    c2w_bcell = tf.nn.rnn_cell.BasicLSTMCell(c2w_single_size, forget_bias=0.0)
    c2w_wf = tf.Variable(tf.random_uniform([c2w_single_size, word_size], -1.0, 1.0))
    c2w_wb = tf.Variable(tf.random_uniform([c2w_single_size, word_size], -1.0, 1.0))
    c2w_b = tf.Variable(tf.random_uniform([word_size], -1.0, 1.0))
    char_vec_flat = tf.reshape(char_vec, [batch_size*max_sentence_length, max_word_length, char_size])
    l_char_flat = tf.reshape(l_char, [batch_size*max_sentence_length])
    c2w_fboutput, c2w_final_state = tf.nn.bidirectional_dynamic_rnn(c2w_fcell, c2w_bcell, char_vec_flat, dtype=tf.float32, sequence_length=l_char_flat)
    c2w_final_forward_state = c2w_final_state[0]
    c2w_foutput = c2w_final_forward_state[1]
    c2w_boutput = tf.reshape(tf.reshape(c2w_fboutput[1], [batch_size, max_sentence_length, max_word_length, c2w_single_size])[:,:,0,:], [batch_size*max_sentence_length, c2w_single_size])
    c2w_output = tf.reshape(tf.matmul(c2w_foutput, c2w_wf) + tf.matmul(c2w_boutput, c2w_wb) + c2w_b, [batch_size, max_sentence_length, word_size])
word_vec = c2w_output

# encode
with tf.variable_scope("encoder"):
    encode_fcell = tf.nn.rnn_cell.BasicLSTMCell(encode_single_size, forget_bias=0.0)
    encode_bcell = tf.nn.rnn_cell.BasicLSTMCell(encode_single_size, forget_bias=0.0)
    encode_wf = tf.Variable(tf.random_uniform([encode_single_size, encode_size], -1.0, 1.0))
    encode_wb = tf.Variable(tf.random_uniform([encode_single_size, encode_size], -1.0, 1.0))
    encode_b = tf.Variable(tf.random_uniform([encode_size], -1.0, 1.0))
    encode_fboutput, encode_final_state = tf.nn.bidirectional_dynamic_rnn(encode_fcell, encode_bcell, word_vec, dtype=tf.float32, sequence_length=l_word)
    encode_final_forward_state = encode_final_state[0]
    encode_foutput = encode_final_forward_state[1]
    encode_boutput = encode_fboutput[1][:,0,:]
    encode_output = tf.matmul(encode_foutput, encode_wf) + tf.matmul(encode_boutput, encode_wb) + encode_b

# LanguageModel
W_label_repr = tf.Variable(tf.random_uniform([num_label, label_repr_size], -1.0, 1.0))
outputs = []
with tf.variable_scope("lm") as scope:
    lm_cell = tf.nn.rnn_cell.BasicLSTMCell(lm_size, forget_bias=0.0)
    lm_w = tf.Variable(tf.random_uniform([lm_size, num_label], -1.0, 1.0))
    lm_b = tf.Variable(tf.random_uniform([num_label], -1.0, 1.0))
    if usePWLoss:
        pass
        # scores = tf.matmul(output, lm_w) + lm_b
        # top_k = tf.nn.top_k(scores, k=beam_width)
        # for b in range(beam_width):
        #     rank_b = tf.transpose(top_k[1])[b,:]
        #     rank_b_embed = tf.nn.embedding_lookup(W_label_repr, rank_b)
        #     scope.reuse_variables()
        #     rank_b_output, rank_b_state = lm_cell(rank_b_embed, new_state)
        #     rank_b_score = tf.matmul(rank_b_output, lm_w) + lm_b
    else:
        lm_outputs = []
        state = lm_cell.zero_state(batch_size, dtype=tf.float32)
        lm_output, state = lm_cell(encode_output, state)
        lm_outputs.append(lm_output)
        for step in range(1, max_sentence_length):
            scope.reuse_variables()
            step_gold_input = tf.nn.embedding_lookup(W_label_repr, gold[:,step])
            previous_top = tf.reshape(tf.nn.top_k(lm_output)[1], [batch_size])
            step_top_input = tf.nn.embedding_lookup(W_label_repr, previous_top)
            step_input = tf.cond(is_train_op, lambda:step_gold_input, lambda:step_top_input)
            lm_output, state = lm_cell(step_input, state)
            lm_outputs.append(lm_output)
        lm_output = tf.pack(lm_outputs, axis=1)
        lm_output_flat = tf.reshape(lm_output, shape=[batch_size*max_sentence_length, lm_size])
        logits_flat = tf.matmul(lm_output_flat, lm_w) + lm_b
        logits = tf.reshape(logits_flat, [batch_size, max_sentence_length, num_label])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, gold)
        # predicts.append(tf.argmax(logits, 1))
        average_loss = tf.reduce_sum(loss) / tf.to_float(tf.reduce_sum(l_word))

print("make train_op")
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(average_loss)

print("make session")
with tf.Session() as sess:
    print("initialize")
    sess.run(tf.initialize_all_variables())

    print("run")
    train_step_end = int(np.ceil(len(train_data.examples) / float(batch_size)))
    dev_step_end = int(np.ceil(len(dev_data.examples) / float(batch_size)))
    for it in range(10):
        for step in range(train_step_end):
            sample_x, sample_l_char, sample_l_word, sample_gold = train_data.RandomSample(batch_size)
            fd = {x:sample_x, l_char:sample_l_char, l_word:sample_l_word, gold:sample_gold, is_train_op:True}
            l, _ = sess.run([average_loss, train_op], feed_dict=fd)
            if step % (train_step_end // 10) == 0: print(l)
        # show dev score
        ls = []
        for step in range(dev_step_end):
            sample_x, sample_l_char, sample_l_word, sample_gold = dev_data.RandomSample(batch_size)
            fd = {x:sample_x, l_char:sample_l_char, l_word:sample_l_word, gold:sample_gold, is_train_op:False}
            l = sess.run(average_loss, feed_dict=fd)
            ls.append(l)
        print("iter %3d   dev loss:%.5f" % (it, sum(ls)/len(ls)))


