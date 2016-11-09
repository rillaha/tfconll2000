import numpy as np
import sys, os, random
from indexer import *

class ConllDataset:
    def __init__(self, filename, indexer, max_sentence_length=78, max_word_length=60):
        self.filename = filename
        with open(filename) as f:
            self.row = f.read().strip()
        subset = []
        self.examples = []
        for line in self.row.split("\n"):
            if len(line) == 0:
                if len(subset) != 0:
                    self.examples.append(subset)
                    subset = []
            else:
                subset.append(line.split(" "))
        if len(subset) != 0:
            self.examples.append(subset)
            subset = []

        self.encoded_examples = []
        for e in self.examples:
            subset = []
            for word in e:
                encoded_word = [[], -1, -1]
                for char in word[0]:
                    encoded_word[0].append(indexer.ConvertToID(char,"char",useUNK=False))
                encoded_word[1] = indexer.ConvertToID(word[1], "POS", useUNK=False)
                encoded_word[2] = indexer.ConvertToID(word[2], "chunk", useUNK=False)
                subset.append(encoded_word)
            self.encoded_examples.append(subset)

        self.end_position = len(self.encoded_examples)
        self.position = self.end_position
        self.order = [i for i in range(self.end_position)]

        self.x = np.zeros((self.end_position, max_sentence_length, max_word_length), dtype=np.int32)
        self.l_char = np.zeros((self.end_position, max_sentence_length), dtype=np.int32)
        self.l_word = np.zeros((self.end_position), dtype=np.int32)
        self.gold = np.zeros((self.end_position, max_sentence_length), dtype=np.int32)
        for s in range(self.end_position):
            esentence = self.encoded_examples[s]
            for w in range(len(esentence)):
                eword = esentence[w]
                self.x[s,w,:len(eword[0])] = eword[0]
                self.l_char[s,w] = len(eword[0])
                self.gold[s,w] = eword[2]
            self.l_word[s] = len(esentence)


    def RandomSample(self, batch_size):
        if self.position == self.end_position:
            random.shuffle(self.order)
            self.position = 0
        if self.position+batch_size <= self.end_position:
            order = self.order[self.position:self.position+batch_size]
            self.position += batch_size
            return self.x[order], self.l_char[order], self.l_word[order], self.gold[order]
        else:
            order = self.order[self.position:]
            remain = batch_size - (self.end_position-self.position)
            self.position = self.end_position
            dammy_x = np.zeros((remain, *self.x.shape[1:]), dtype=np.int32)
            ret_x = np.r_[self.x[order], dammy_x]
            dammy_l_char = np.zeros((remain, *self.l_char.shape[1:]), dtype=np.int32)
            ret_l_char = np.r_[self.l_char[order], dammy_l_char]
            dammy_l_word = np.zeros(remain, dtype=np.int32)
            ret_l_word = np.r_[self.l_word[order], dammy_l_word]
            dammy_gold = np.zeros((remain, *self.gold.shape[1:]), dtype=np.int32)
            ret_gold = np.r_[self.gold[order], dammy_gold]
            return ret_x, ret_l_char, ret_l_word, ret_gold

