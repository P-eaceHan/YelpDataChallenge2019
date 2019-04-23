import numpy as np
# from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.initializers import Constant
from keras.layers import Input, Dense, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

# getting the pre-trained word embeddings
path = '/home/peace/edu/3/'
filename = 'model.txt'  # NLPL dataset 3
print("Indexing word vectors from", filename)

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
labelfile = 'review_labels.tsv'

features = []  # list of features
label_index = {}  # dictionary of label name, numeric id
labels = []  # list of star ratings
maxlength = 0
with open(path+filename) as f:
    for line in f:
        line = line.split()
        if len(line) > maxlength:
            maxlength = len(line)
        revid = line[0]
        star = int(float(line[1]))
        # pos = line[2]
        # neg = line[3]
        label_id = len(label_index)
        label_index[str(star)] = star
        features.append(line[2:])
        labels.append(star)
print("total number of features:", len(features))
print("total number of labels:", len(labels))
print("max feature_vec length:", maxlength)
# store labels as set
label_set = set(labels)

# vectorize text features into 2D integer tensor
MAX_NUM_WORDS = 25000
MAX_SEQUENCE_LENGTH = 1000

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(features)
sequences = tokenizer.texts_to_sequences(features)

word_index = tokenizer.word_index
print("Found {} unique tokens".format(len(word_index)))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


def to_cat_tensor(orig_vec, n_cls):
    batch_size, n_rows, n_cols = orig_vec.shape
    x1d = orig_vec.ravel()
    y1d = to_categorical(x1d, num_classes=n_cls)
    y4d = y1d.reshape([batch_size, n_rows, n_cols, n_cls])
    return y4d


labels = to_categorical(np.asarray(labels))
print(labels)
print("Shape of data tensor: ", data.shape)
print("Shape of label tensor: ", labels.shape)

# splitting into training and validation
VALIDATION_SPLIT = 0.2
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
y_train = to_cat_tensor(y_train, 5)
x_val = data[-num_validation_samples]
y_val = labels[-num_validation_samples]
y_val = to_cat_tensor(y_val, 5)


# prepping the embedding matrix
print("Preparing embedding matrix...")

EMBEDDING_DIM = 100
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
print(len(word_index)+1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


# training the model on the data
print("Training model...")
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
print("layer number {} trained.".format(1))
x = MaxPooling1D(5)(x)
print("layer number {} trained.".format(2))
x = Conv1D(128, 5, activation='relu')(x)
print("layer number {} trained.".format(3))
x = MaxPooling1D(5)(x)
print("layer number {} trained.".format(4))
x = Conv1D(128, 5, activation='relu')(x)
print("layer number {} trained.".format(5))
x = GlobalMaxPooling1D()(x)
print("layer number {} trained.".format(6))
x = Dense(128, activation='relu')(x)
print("layer number {} trained.".format(7))
preds = Dense(len(label_index), activation='softmax')(x)
print(len(label_index))
print("Final dense layer trained and preds calculated.")
print("Shape of sequence_input: ", sequence_input.shape)
print("Shape of preds: ", preds.shape)
# TODO: the shape of preds? or labels? is wrong...
# Error when checking target: expected dense_2 to have shape (5,) but got array with shape (6,)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))


