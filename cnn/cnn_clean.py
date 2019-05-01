import time
import numpy as np
import pickle
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


# output log file
o_file = 'exp5.txt'
logfile = open(o_file, 'w')

# getting the features file
# path = '../data/task2/'
path = '../cnn_data/'
filename = 'rawText_lemmatized.tsv'  # the lemmatized review texts
# filename = 'jjOnly.tsv'            # adjectives and contexts only
# filename = 'nnOnly.tsv'            # nouns and contexts only
# filename = 'jnMix.tsv'             # adjectives and nouns and contexts, in order
# filename = 'jnSep.tsv'             # adjectives and nouns and contexts, separated
# filename = 'test_features.tsv'     # small sample file for testing
logfile.write("results for: " + filename + '\n\n')
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
logfile.write("total number of reviews:")
logfile.write(str(len(reviews)))
logfile.write('\n')
logfile.write("total number of labels:")
logfile.write(str(len(labels)))
logfile.write('\n')
print("max feature_vec length:", maxlength)
# store labels as set
label_set = set(labels)
print(reviews[0])

# vectorize text features into 2D integer tensor
MAX_NUM_WORDS = 5000
MAX_SEQUENCE_LENGTH = 1000

start_time = time.time()
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)  # TODO: try without setting max limit?
tokenizer.fit_on_texts(reviews)
end_time = np.round(time.time() - start_time, 2)
print("tokenizer fit on reviews")
print("Fitting time:", end_time)
logfile.write("Fitting time:")
logfile.write(str(end_time))
logfile.write('s\n')

sequences = tokenizer.texts_to_sequences(reviews)
print('Max seq len', MAX_SEQUENCE_LENGTH)
end_time = np.round(time.time() - start_time, 2)
print("model fitted on x_train, y_train")
print("Training time:", end_time)

word_index = tokenizer.word_index
MAX_NUM_WORDS = len(word_index) + 1
print('max num words (vocab size)', MAX_NUM_WORDS)
logfile.write("max num words (vocab size)")
logfile.write(str(MAX_NUM_WORDS))
logfile.write('\n')
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

labels = to_categorical(np.asarray(labels))
print("Shape of data tensor: ", data.shape)
print("Shape of label tensor: ", labels.shape)

emb_pickle = open('embeddings_index.pkl', 'rb')
embeddings_index = pickle.load(emb_pickle)
emb_pickle.close()

# splitting into training and validation
VALIDATION_SPLIT = 0.2
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
print("Shape of x_train: ", x_train.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of x_val: ", x_val.shape)
print("Shape of y_val: ", y_val.shape)
logfile.write("Shape of x_train: ")
logfile.write(str(x_train.shape))
logfile.write('\n')
logfile.write("Shape of y_train: ")
logfile.write(str(y_train.shape))
logfile.write('\n')
logfile.write("Shape of x_val: ")
logfile.write(str(x_val.shape))
logfile.write('\n')
logfile.write("Shape of y_val: ")
logfile.write(str(y_val.shape))
logfile.write('\n')

EMBEDDING_DIM = 300
num_words = len(word_index) + 1

num_words = len(word_index) + 1
print("vocab_size:", len(word_index)+1)
logfile.write("vocab size: ")
logfile.write(str(len(word_index)+1))
logfile.write('\n')
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > num_words - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
nonzero_elems = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print("word embedding coverage:", nonzero_elems / num_words)
logfile.write("word embedding coverage: ")
logfile.write(str(nonzero_elems / num_words))
logfile.write('\n')

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
# model.add(Conv1D(3, 5, activation='relu'))
# model.add(MaxPooling1D(5))
# model.add(Flatten())
# model.add(Dense(5, activation='softmax'))

# exp4 structure
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1000, activation='relu'))
model.add(Dense(y_train.shape[1], activation='sigmoid'))  # TODO: next run, change to 'softmax' for lemmas

tensorBoard = TensorBoard(log_dir='./logs', write_graph=True)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc', 'mse', 'cosine'],)
print("Model Summary:")
logfile.write("Model Summary:" + '\n')
model.summary()  # need to manually past summary to logfile after
logfile.write('\n')
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
logfile.write("Training time: " + end_time)
logfile.write('\n')
logfile.write('\n')

print("Evaluating the model...")
start_time = time.time()
score, acc, mse_me, cosine = model.evaluate(x_val, y_val, verbose=1)
end_time = np.round(time.time() - start_time, 2)
print("Eval time:", end_time)
logfile.write("Eval time: " + end_time)
logfile.write('\n')

logfile.write("score:" + score)
logfile.write('\n')
logfile.write("acc:" + acc)
logfile.write('\n')
logfile.write("mse:" + mse_me)
logfile.write('\n')
logfile.write("cosine:" + cosine)
logfile.write('\n')
logfile.write('\n')

# Evaluating TEST
logfile.write("Evaluating TEST model class prediction")
logfile.write('\n')
start_time = time.time()
y_pred = model.predict(x_val, 128, verbose=2)
end_time = np.round(time.time() - start_time, 2)
logfile.write("predict time:" + end_time)
logfile.write('\n')
logfile.write(classification_report(y_val.argmax(axis=1),
                                    y_pred.argmax(axis=1)))
logfile.write('\n')
mse = mean_squared_error(y_val, y_pred)
logfile.write("mean squared error:" + mse)
logfile.write('\n')
logfile.write("RMSE:" + sqrt(mse))
logfile.write('\n')
r2 = r2_score(y_val, y_pred)
logfile.write("r2:" + r2)
logfile.write('\n')
logfile.write('\n')
# Evaluating TRAIN
logfile.write("Evaluating TRAIN model class prediction")
logfile.write('\n')
start_time = time.time()
y_pred = model.predict(x_train, 128, verbose=2)
end_time = np.round(time.time() - start_time, 2)
logfile.write("predict time:" + end_time)
logfile.write('\n')
logfile.write(classification_report(y_train.argmax(axis=1),
                                    y_pred.argmax(axis=1)))
logfile.write('\n')
mse = mean_squared_error(y_train, y_pred)
logfile.write("mean squared error:" + mse)
logfile.write('\n')
logfile.write("RMSE:" + sqrt(mse))
logfile.write('\n')
r2 = r2_score(y_train + y_pred)
logfile.write("r2:" + r2)
logfile.write('\n')

logfile.close()

