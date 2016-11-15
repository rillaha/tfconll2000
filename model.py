#!/usr/bin/python3
import sys
import tensorflow as tf
import numpy as np
import conllreader

np.random.seed(0)
tf.set_random_seed(0)

class Model:
    def __init__(self):
        self.usePWLoss = False
        self.useStructuredLearning = False
        # self.SetParameters()
        self.SetMinimalParameters()

    def Run(self, sess):
        self.ReadDataset()
        self.MakeGraph()
        self.MakeTrainOp()
        print("initialize")
        sess.run(tf.initialize_all_variables())

        print("run")
        for it in range(10):
            print("train:")
            train_loss, train_acc = self.Train(sess)
            print("iter:%d  loss:%.5f acc:%.5f" % (it, train_loss, train_acc))
            print("devel:")
            dev_loss, dev_acc = self.Dev(sess)
            print("iter:%d  loss:%.5f acc:%.5f" % (it, dev_loss, dev_acc))
        return

    def SetParameters(self):
        self.batch_size = 32
        self.max_sentence_length = 78
        self.max_word_length = 60
        self.char_size = 50
        self.char_vocab = 256
        self.c2w_single_size = 100
        self.word_size = 150
        self.encode_single_size = 100
        self.encode_size = 100
        self.lm_size = 100
        self.num_label = 24
        self.label_repr_size = self.encode_size
        self.beam_width = 10
        return

    def SetMinimalParameters(self):
        self.batch_size = 128
        self.max_sentence_length = 78
        self.max_word_length = 60
        self.char_size = 2
        self.char_vocab = 256
        self.c2w_single_size = 3
        self.word_size = 5
        self.encode_single_size = 5
        self.encode_size = 2
        self.lm_size = 3
        self.num_label = 24
        self.label_repr_size = self.encode_size
        self.beam_width = 10
        return

    def ReadDataset(self):
        print("read dataset", end="")
        sys.stdout.flush()
        self.indexer = conllreader.Indexer()
        print(".", end="")
        sys.stdout.flush()
        self.train_data = conllreader.ConllDataset("corpus/train.head.txt", self.indexer)
        print(".", end="")
        sys.stdout.flush()
        self.dev_data = conllreader.ConllDataset("corpus/train.tail.txt", self.indexer)
        print(".", end="")
        sys.stdout.flush()
        self.test_data = conllreader.ConllDataset("corpus/test.txt", self.indexer)
        print(".", end="")
        sys.stdout.flush()
        self.indexer.update()
        self.indexer.close()

        self.num_train_word = np.sum(self.train_data.l_word)
        self.num_dev_word = np.sum(self.dev_data.l_word)
        print()
        return

    def MakeGraph(self):
        print("make graph")
        self.is_train_op = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.int32, [self.batch_size, self.max_sentence_length, self.max_word_length])
        self.l_char = tf.placeholder(tf.int32, [self.batch_size, self.max_sentence_length])
        self.l_word = tf.placeholder(tf.int32, [self.batch_size])
        self.gold = tf.placeholder(tf.int32, [self.batch_size, self.max_sentence_length])

        self.W_char_emb = tf.Variable(tf.random_uniform([self.char_vocab, self.char_size], -1.0, 1.0))
        self.char_vec = tf.nn.embedding_lookup(self.W_char_emb, self.x)

        # c2w
        with tf.variable_scope("c2w"):
            self.char_vec_flat = tf.reshape(self.char_vec, [self.batch_size*self.max_sentence_length, self.max_word_length, self.char_size])
            self.l_char_flat = tf.reshape(self.l_char, [self.batch_size*self.max_sentence_length])
            self.c2w_fcell = tf.nn.rnn_cell.BasicLSTMCell(self.c2w_single_size, forget_bias=0.0)
            self.c2w_bcell = tf.nn.rnn_cell.BasicLSTMCell(self.c2w_single_size, forget_bias=0.0)
            self.c2w_wf = tf.Variable(tf.random_uniform([self.c2w_single_size, self.word_size], -1.0, 1.0))
            self.c2w_wb = tf.Variable(tf.random_uniform([self.c2w_single_size, self.word_size], -1.0, 1.0))
            self.c2w_b = tf.Variable(tf.random_uniform([self.word_size], -1.0, 1.0))
            self.c2w_fboutput, self.c2w_final_state = tf.nn.bidirectional_dynamic_rnn(self.c2w_fcell, self.c2w_bcell, self.char_vec_flat, dtype=tf.float32, sequence_length=self.l_char_flat)
            self.c2w_final_forward_state = self.c2w_final_state[0]
            self.c2w_foutput = self.c2w_final_forward_state[1]
            self.c2w_boutput = self.c2w_fboutput[1][:,0,:]
            self.c2w_output_flat = tf.matmul(self.c2w_foutput, self.c2w_wf) + tf.matmul(self.c2w_boutput, self.c2w_wb) + self.c2w_b
            self.c2w_output = tf.reshape(self.c2w_output_flat, [self.batch_size, self.max_sentence_length, self.word_size])
        self.word_vec = self.c2w_output

        # encode
        with tf.variable_scope("encoder"):
            self.encode_fcell = tf.nn.rnn_cell.BasicLSTMCell(self.encode_single_size, forget_bias=0.0)
            self.encode_bcell = tf.nn.rnn_cell.BasicLSTMCell(self.encode_single_size, forget_bias=0.0)
            self.encode_wf = tf.Variable(tf.random_uniform([self.encode_single_size, self.encode_size], -1.0, 1.0))
            self.encode_wb = tf.Variable(tf.random_uniform([self.encode_single_size, self.encode_size], -1.0, 1.0))
            self.encode_b = tf.Variable(tf.random_uniform([self.encode_size], -1.0, 1.0))
            self.encode_fboutput, self.encode_final_state = tf.nn.bidirectional_dynamic_rnn(self.encode_fcell, self.encode_bcell, self.word_vec, dtype=tf.float32, sequence_length=self.l_word)
            self.encode_final_forward_state = self.encode_final_state[0]
            self.encode_foutput = self.encode_final_forward_state[1]
            self.encode_boutput = self.encode_fboutput[1][:,0,:]
            self.encode_output = tf.matmul(self.encode_foutput, self.encode_wf) + tf.matmul(self.encode_boutput, self.encode_wb) + self.encode_b

        # PWLoss
        def NormalizedDistance(batch_vec, lookup_table):
            batch_size = batch_vec.get_shape()[0].value
            dim = batch_vec.get_shape()[1].value
            nvec = tf.nn.l2_normalize(batch_vec, 1)
            ntable = tf.nn.l2_normalize(lookup_table, 1)
            diffs = ntable - tf.reshape(nvec, [batch_size, 1, dim])
            distances = tf.reduce_sum(tf.square(diffs), 2)
            # shape(distances)=[batch_size, num_of_lookup_vecs]
            return distances

        # LanguageModel
        self.W_input_label_repr = tf.Variable(tf.random_uniform([self.num_label+1, self.label_repr_size], -1.0, 1.0))
        self.W_output_label_repr = tf.Variable(tf.random_uniform([self.num_label+1, self.label_repr_size], -1.0, 1.0))
        self.outputs = []
        with tf.variable_scope("lm") as scope:
            self.lm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lm_size, forget_bias=0.0)
            self.lm_w = tf.Variable(tf.random_uniform([self.lm_size, self.num_label], -1.0, 1.0))
            self.lm_b = tf.Variable(tf.random_uniform([self.num_label], -1.0, 1.0))
            if self.usePWLoss:
                pass
                # scores = tf.matmul(output, lm_w) + lm_b
                # top_k = tf.nn.top_k(scores, k=beam_width)
                # for b in range(beam_width):
                #     rank_b = tf.transpose(top_k[1])[b,:]
                #     rank_b_embed = tf.nn.embedding_lookup(W_input_label_repr, rank_b)
                #     scope.reuse_variables()
                #     rank_b_output, rank_b_state = lm_cell(rank_b_embed, new_state)
                #     rank_b_score = tf.matmul(rank_b_output, lm_w) + lm_b
            else:
                self.logits_list = []
                state = self.lm_cell.zero_state(self.batch_size, dtype=tf.float32)
                lm_step_output, state = self.lm_cell(self.encode_output, state)
                step_logits = tf.matmul(lm_step_output, self.lm_w) + self.lm_b
                self.logits_list.append(step_logits)
                for step in range(1, self.max_sentence_length):
                    scope.reuse_variables()
                    step_gold_input = tf.nn.embedding_lookup(self.W_input_label_repr, self.gold[:,step])
                    previous_top = tf.reshape(tf.nn.top_k(step_logits)[1], [self.batch_size])
                    step_top_input = tf.nn.embedding_lookup(self.W_input_label_repr, previous_top)
                    step_input = tf.cond(self.is_train_op, lambda:step_gold_input, lambda:step_top_input)
                    lm_step_output, state = self.lm_cell(step_input, state)
                    step_logits = tf.matmul(lm_step_output, self.lm_w) + self.lm_b
                    self.logits_list.append(step_logits)
                # shape(logits)=(self.batch_size, self.max_sentence_length, self.num_label)
                self.logits = tf.pack(self.logits_list, axis=1)
                mask = tf.sequence_mask(self.l_word, self.max_sentence_length, dtype=tf.float32)
                self.word_count = tf.to_float(tf.reduce_sum(self.l_word))
                self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.gold) * mask)
                self.average_loss = self.loss / self.word_count
                self.predict = tf.to_int32(tf.argmax(self.logits, dimension=2))
                self.correct = tf.reduce_sum(tf.to_float(tf.equal(self.predict, self.gold)) * mask)
                self.accuracy = self.correct / self.word_count
        return

    def MakeTrainOp(self):
        print("make train_op")
        self.train_op = tf.train.GradientDescentOptimizer(0.1).minimize(self.average_loss)
        return

    def Train(self, sess):
        train_loss = 0.0
        train_acc = 0.0
        train_step_end = int(np.ceil(len(self.train_data.examples) / float(self.batch_size)))
        for step in range(train_step_end):
            sample_x, sample_l_char, sample_l_word, sample_gold = self.train_data.RandomSample(self.batch_size)
            fd = {self.x:sample_x, self.l_char:sample_l_char, self.l_word:sample_l_word, self.gold:sample_gold, self.is_train_op:True}
            al, l, c, _ = sess.run([self.average_loss, self.loss, self.correct, self.train_op], feed_dict=fd)
            train_loss += l
            train_acc += c
            if step % (train_step_end // 10) == 0: print(al)
        train_loss /= self.num_train_word
        train_acc /= self.num_train_word
        return train_loss, train_acc

    def Dev(self, sess):
        dev_loss = 0.0
        dev_acc = 0.0
        dev_step_end = int(np.ceil(len(self.dev_data.examples) / float(self.batch_size)))
        for step in range(dev_step_end):
            sample_x, sample_l_char, sample_l_word, sample_gold = self.dev_data.RandomSample(self.batch_size)
            fd = {self.x:sample_x, self.l_char:sample_l_char, self.l_word:sample_l_word, self.gold:sample_gold, self.is_train_op:True}
            l, step_correct = sess.run([self.loss, self.correct], feed_dict=fd)
            dev_loss += l
            dev_acc += step_correct
        dev_loss /= self.num_dev_word
        dev_acc /= self.num_dev_word
        return dev_loss, dev_acc

if __name__ == "__main__":
    model = Model()
    with tf.Session() as s:
        model.Run(s)



