mkdir -p corpus
wget http://www.cnts.ua.ac.be/conll2000/chunking/train.txt.gz -O corpus/train.txt.gz
wget http://www.cnts.ua.ac.be/conll2000/chunking/test.txt.gz -O corpus/test.txt.gz
gunzip corpus/train.txt.gz corpus/test.txt.gz
head -n 200084 corpus/train.txt > corpus/train.head.txt
tail -n 20579 corpus/train.txt > corpus/train.tail.txt
