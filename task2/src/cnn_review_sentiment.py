import numpy as np
from keras.layers.embeddings import Embedding


# getting the pre-trained word embeddings
path = '/home/peace/edu/3/'
filename = 'model.txt'  # NLPL dataset 3
print("Indexing word vectors from ", filename)

embeddings_index = {}
with open(path + filename) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

print("Found {} word vectors.".format(len(embeddings_index)))


# getting the features file
path = '../data/task2/'
filename = 'review_features.tsv'


