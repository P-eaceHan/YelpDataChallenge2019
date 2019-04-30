import pickle
import time
import numpy as np

# prepping the embedding matrix
print("Preparing embedding matrix...")
# getting the pre-trained word embeddings
# path = '/home/peace/edu/3/'
path = '../../3/'
filename = 'model.txt'  # NLPL dataset 3
# download from http://vectors.nlpl.eu/repository/ (search for English)
# ID 3, vector size 300, window 5 'English Wikipedia Dump of February 2017'
# vocab size: 296630; Algo: Gensim Continuous Skipgram; Lemma: True
print("Indexing word vectors from", filename)
start_time = time.time()
embeddings_index = {}
with open(path + filename) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        word = word.split('_')[0]  # get just the word, not POS info
        embeddings_index[word] = coefs

embeddings_file = open('embeddings_index.pkl', 'wb')
pickle.dump(embeddings_index, embeddings_file)
embeddings_file.close()
end_time = np.round(time.time() - start_time, 2)
print("Found {} word vectors.".format(len(embeddings_index)))
print("Time to fetch and save word embeddings: {}s".format(end_time))

