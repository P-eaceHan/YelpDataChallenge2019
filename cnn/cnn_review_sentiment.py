import numpy as np
# from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.initializers import Constant
from keras.layers import Input, Dense, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten, Dropout
from keras.models import Model, Sequential
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import TensorBoard


# getting the pre-trained word embeddings
path = '/home/peace/edu/3/'
filename = 'model.txt'  # NLPL dataset 3
# download from http://vectors.nlpl.eu/repository/ (search for English)
# ID 3, vector size 300, window 5 'English Wikipedia Dump of February 2017'
# vocab size: 296630; Algo: Gensim Continuous Skipgram; Lemma: True
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
        star = int(float(line[1])) - 1
        # pos = line[2]
        # neg = line[3]
        label_id = len(label_index)
        label_index[str(star)] = star + 1
        features.append(line[2:])
        labels.append(star)
print("total number of features:", len(features))
print("total number of labels:", len(labels))
print("max feature_vec length:", maxlength)
# store labels as set
label_set = set(labels)
print(features[0])
# vectorize text features into 2D integer tensor
MAX_NUM_WORDS = 2500
MAX_SEQUENCE_LENGTH = 1000

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(features)
sequences = tokenizer.texts_to_sequences(features)

word_index = tokenizer.word_index
print("Found {} unique tokens".format(len(word_index)))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

# print(labels)
# print(label_set)
# labels = np.asarray(labels)
# label_set = set(labels)
# print(labels)
# print(label_set)
labels = to_categorical(np.asarray(labels))
# labels = to_categorical(labels)
# encoder = LabelEncoder()
# encoder.fit(labels)
# labels = encoder.transform(labels)
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
print("Shape of data tensor: ", data.shape)
print("Shape of label tensor: ", labels.shape)

x_train = data[:-num_validation_samples]
print(x_train[0])
y_train = labels[:-num_validation_samples]
# y_train = to_cat_tensor(y_train, 5)
x_val = data[-num_validation_samples]
y_val = labels[-num_validation_samples]
# y_val = to_cat_tensor(y_val, 5)
print("Shape of x_train: ", x_train.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of x_val: ", x_val.shape)
print("Shape of y_val: ", y_val.shape)


# prepping the embedding matrix
print("Preparing embedding matrix...")

EMBEDDING_DIM = 100
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
print(len(word_index)+1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    # if i > MAX_NUM_WORDS:
    if i > num_words - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
# model = Sequential()
# model.add(embedding_layer)
# model.add(Conv1D(64, 3, padding='same'))
# # model.add(MaxPooling1D(5))
# model.add(Conv1D(32, 3, padding='same'))
# # model.add(MaxPooling1D(5))
# model.add(Conv1D(16, 3, padding='same'))
# model.add(Flatten())
# model.add(Dropout(0.2))
# # model.add(GlobalMaxPooling1D())
# model.add(Dense(180, activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(len(label_index), activation='softmax'))

tensorBoard = TensorBoard(log_dir='./logs', write_graph=True)

# training the model on the data
print("Training model...")
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
print(embedded_sequences.shape)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
print("layer number {} trained.".format(1))
x = MaxPooling1D(5)(x)
print("layer number {} trained.".format(2))
x = Conv1D(128, 5, activation='relu')(x)
print("layer number {} trained.".format(3))
# x = MaxPooling1D(5)(x)
# print("layer number {} trained.".format(4))
# x = Conv1D(128, 5, activation='relu')(x)
# print("layer number {} trained.".format(5))
# x = GlobalMaxPooling1D()(x)
x = MaxPooling1D(35)(x)
print("layer number {} trained.".format(6))
x = Flatten()(x)
print("layer flatten trained")
x = Dense(128, activation='relu')(x)
print("layer number {} trained.".format(7))
preds = Dense(len(label_index), activation='softmax')(x)
print(len(label_index))
print("Final dense layer trained and preds calculated.")
print("Shape of sequence_input: ", sequence_input.shape)
print("Shape of preds: ", preds.shape)
# TODO: the shape of features is wrong...?
# Error when checking target: expected dense_2 to have shape (5,) but got array with shape (6,)
model = Model(sequence_input, preds)
print("Model created", model)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'], verbose=2)
print("model compiled successfully")
model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val), verbose=2,
          callbacks=[tensorBoard])
print("model fitted on {}, {}".format(x_train, y_train))
