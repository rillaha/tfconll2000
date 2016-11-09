import conllreader as cr

indexer = cr.Indexer()
print("read train")
train_data = cr.ConllDataset("corpus/train.head.txt", indexer)
print("read dev")
dev_data = cr.ConllDataset("corpus/train.tail.txt", indexer)
print("read test")
test_data = cr.ConllDataset("corpus/test.txt", indexer)
print("update")
indexer.update()
print("len char:%d  len POS:%d  len chunk:%d" % (len(indexer.dic["char"]), len(indexer.dic["POS"]), len(indexer.dic["chunk"])))

