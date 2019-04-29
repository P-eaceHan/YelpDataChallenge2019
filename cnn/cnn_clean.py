import time
import numpy as np
from math import sqrt
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.initializers import Constant
from keras.layers import Input, Dense, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten, Dropout
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard
from sklearn.metrics import classification_report, mean_squared_error, r2_score

# getting the features file
# path = '../data/task2/'
path = '../'
# filename = 'review_features.tsv'
# # labelfile = 'review_labels.tsv'
#
# features = []  # list of features
# label_index = {}  # dictionary of label name, numeric id
# labels = []  # list of star ratings
# maxlength = 0
# with open(path+filename) as f:
#     for line in f:
#         line = line.split()
#         if len(line) > maxlength:
#             maxlength = len(line)
#         revid = line[0]
#         star = int(float(line[1])) - 1
#         # pos = line[2]
#         # neg = line[3]
#         label_id = len(label_index)
#         label_index[str(star)] = star + 1
#         features.append(line[2:])
#         labels.append(star)
# print("total number of features:", len(features))
# print("total number of labels:", len(labels))
# print("max feature_vec length:", maxlength)

filename = 'review_features2.tsv'
# filename = 'test_features.tsv'
reviews = []  # list of reviews
label_index = {}  # dictionary of label name, numeric id
labels = []  # lThe ist of star ratings
maxlength = 0
with open(path + filename) as f:
    for line in f:
        line = line.split('\t')
        # print(line)
        text = line[2].strip()
        # text = text_to_word_sequence(text)
        if len(text) > maxlength:
            maxlength = len(text)
        revid = line[0]
        star = int(float(line[1])) - 1
        label_id = len(label_index)  # ?
        label_index[str(star)] = star + 1
        reviews.append(text)
        labels.append(star)
print("total number of reviews:", len(reviews))
print("total number of labels:", len(labels))
print("max feature_vec length:", maxlength)
# store labels as set
label_set = set(labels)
print(reviews[0])

# vectorize text features into 2D integer tensor
# MAX_NUM_WORDS = 5000
MAX_SEQUENCE_LENGTH = 1000

# tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(reviews)

sequences = tokenizer.texts_to_sequences(reviews)
# MAX_SEQUENCE_LENGTH = max([len(s.split()) for s in reviews])
print('Max seq len', MAX_SEQUENCE_LENGTH)

word_index = tokenizer.word_index
MAX_NUM_WORDS = len(word_index) + 1
print('max num words (vocab size)', MAX_NUM_WORDS)
# print("Found {} unique tokens".format(len(word_index)))
# print(sequences[0])
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

labels = to_categorical(np.asarray(labels))
# labels = to_categorical(labels)
# encoder = LabelEncoder()
# encoder.fit(labels)
# labels = encoder.transform(labels)
# print(labels)
print("Shape of data tensor: ", data.shape)
print("Shape of label tensor: ", labels.shape)

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

embeddings_index = {}
with open(path + filename) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        word = word.split('_')[0]  # get just the word, not POS info
        embeddings_index[word] = coefs

print("Found {} word vectors.".format(len(embeddings_index)))

# num_words = min(MAX_NUM_WORDS, len(word_index) + 1)

# splitting into training and validation
VALIDATION_SPLIT = 0.2
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
# print(x_train[0])
y_train = labels[:-num_validation_samples]
# y_train = to_cat_tensor(y_train, 5)
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
# y_val = to_cat_tensor(y_val, 5)
print("Shape of x_train: ", x_train.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of x_val: ", x_val.shape)
print("Shape of y_val: ", y_val.shape)


EMBEDDING_DIM = 300
num_words = len(word_index) + 1

num_words = len(word_index) + 1
print("vocab_size:", len(word_index)+1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    # if i > MAX_NUM_WORDS:
    if i > num_words - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
nonzero_elems = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print("word embedding coverage:", nonzero_elems / num_words)

embedding_layer = Embedding(x_train.shape[1],
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
model = Sequential()
model.add(Embedding(num_words, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
# exp1 structure
# model.add(Conv1D(128, 5, activation='relu'))
# model.add(GlobalMaxPooling1D())
# model.add(Dense(1000, activation='relu'))
# model.add(Dense(5, activation='sigmoid'))

# exp2 structure
# model.add(Conv1D(128, 5, activation='relu'))
# model.add(MaxPooling1D(5))
# model.add(Conv1D(128, 5, activation='relu'))
# model.add(MaxPooling1D(5))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(5, activation='sigmoid'))

# exp3 structure
model.add(Conv1D(3, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Flatten())
model.add(Dense(5, activation='softmax'))

tensorBoard = TensorBoard(log_dir='./logs', write_graph=True)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc', 'mse', 'cosine'],)
print("Model Summary:")
model.summary()
print("model compiled successfully")
print("Training the model...")
start_time = time.time()
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=5,
                    validation_data=(x_val, y_val), verbose=2,
                    callbacks=[tensorBoard])
end_time = np.round(time.time() - start_time, 2)
print("model fitted on x_train, y_train")
print("Training time:", end_time)

print("Evaluating the model...")
start_time = time.time()
score, acc, mse_me, cosine = model.evaluate(x_val, y_val, verbose=1)
end_time = np.round(time.time() - start_time, 2)
print("Eval time:", end_time)

print("score:", score)
print("acc:", acc)
print("mse:", mse_me)
print("cosine:", cosine)

# Evaluating TEST
print("Evaluating TEST model class prediction")
start_time = time.time()
y_pred = model.predict(x_val, 128, verbose=2)
end_time = np.round(time.time() - start_time, 2)
print("predict time:", end_time)
print(classification_report(y_val.argmax(axis=1),
                            y_pred.argmax(axis=1)))
mse = mean_squared_error(y_val, y_pred)
print("mean squared error:", mse)
print("RMSE:", sqrt(mse))
r2 = r2_score(y_val, y_pred)
print("r2:", r2)
print()
# Evaluating TRAIN
print("Evaluating TRAIN model class prediction")
start_time = time.time()
y_pred = model.predict(x_train, 128, verbose=2)
end_time = np.round(time.time() - start_time, 2)
print("predict time:", end_time)
print(classification_report(y_train.argmax(axis=1),
                            y_pred.argmax(axis=1)))
mse = mean_squared_error(y_train, y_pred)
print("mean squared error:", mse)
print("RMSE:", sqrt(mse))
r2 = r2_score(y_train, y_pred)
print("r2:", r2)

