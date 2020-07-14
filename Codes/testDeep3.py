import time
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import SimpleRNN, Embedding, Flatten, Dense, LSTM
from keras import layers
from keras import backend as K
from corpora_utils import CorporaHelper,CorporaDomains, CorporaProperties
import numpy as np
import os
"""
Binary bidirectional LSTM RNN with two layers and 2 emotions
"""
start_time = time.time()
#corpus = 'All this not sleeping has a terrible way of playing with your memory.' # fear => test
#corpus = "The Rock is destined to be the 21st Century s new Conan and that he s going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal ."
corpus = 'If you sometimes like to go to the movies to have fun , Wasabi is a good place to start .' # joy 
# load data
labels = []
texts = []
corpora_helper = CorporaHelper("multigenre.csv")
count_joy = 0
count_sadness = 0
count_anger = 0
count_fear = 0

for index, corpus in corpora_helper.get_data().iterrows():
    # only moviereviews
    if True or corpus[CorporaProperties.DOMAIN.value] == CorporaDomains.MOVIEREVIEW.value:
        # only joy
        # only disgust
        if corpus[CorporaProperties.EMOTION.value] == 'sadness':
            labels.append(0)
            texts.append(corpus[CorporaProperties.CLEANED_CORPUS.value])
            count_sadness += 1
        elif corpus[CorporaProperties.EMOTION.value] == 'joy':
            labels.append(1)
            texts.append(corpus[CorporaProperties.CLEANED_CORPUS.value])
            count_joy += 1
        elif corpus[CorporaProperties.EMOTION.value] == 'anger':
            labels.append(1)
            texts.append(corpus[CorporaProperties.CLEANED_CORPUS.value])
            count_anger += 1
        elif corpus[CorporaProperties.EMOTION.value] == 'fear':
            labels.append(1)
            texts.append(corpus[CorporaProperties.CLEANED_CORPUS.value])
            count_fear += 1
print('number of anger labels: ',count_anger)
print('number of fear labels: ', count_fear)
print('number of joy labels: ',count_joy)
print('number of sadness labels: ', count_sadness)


## Create one hot encoding
maxlen = 100 # max. number of words in sequences
training_samples = 7000
validation_samples = 1000
test_samples = 1000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

#one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
#print(one_hot_results)

word_index = tokenizer.word_index
print ('%s eindeutige Tokens gefunden.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data:', data.shape)
print('Shape of labels:', labels.shape)

# mix the data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# split in train and validate
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]
x_test = data[training_samples + validation_samples: training_samples + validation_samples + test_samples]
y_test = labels[training_samples + validation_samples: training_samples + validation_samples + test_samples]

# Read the glove embedding acc. 6.1.3
glove_dir = './glove.twitter.27B'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.twitter.27B.200d.txt', ), encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# Create embedding matrix
embedding_dim = 200
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

# Create model
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Bidirectional

"""
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.summary()
"""
max_features = max_words

model = Sequential()
#model.add(Embedding(max_features, 32))
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
#model.add(LSTM(32))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(Dense(1, activation='sigmoid'))
model.summary()

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


"""
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten()) #3D to 2D
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
"""

# Load GloVe embedding
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

# Train and Evaluate
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])
history = model.fit(x_train, y_train,
                    epochs=2,
                    batch_size=128,
                    validation_data=(x_val, y_val))
"""
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
"""
model.save_weights('pre_trained_glove_model.h5')

# Test model
print("Evaluate on test data")
model.load_weights('pre_trained_glove_model.h5')
#results = model.evaluate(x_test, y_test, batch_size=128)
loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)
print("Loss:", loss)
print("Accuracy:", accuracy)
print("F1-Score", f1_score)
print("Precision:", precision)
print("Recall:", recall)


# Plot performance
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

elapsed_time = time.time() - start_time
print("Elapsed Time:", elapsed_time)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()



