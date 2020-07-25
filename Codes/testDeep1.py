from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from corpora_utils import CorporaHelper,CorporaDomains, CorporaProperties
import numpy as np
import os
"""
Binary simple NN with two layers and 2 emotions
"""


#corpus = 'All this not sleeping has a terrible way of playing with your memory.' # fear => test
#corpus = "The Rock is destined to be the 21st Century s new Conan and that he s going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal ."
corpus = 'If you sometimes like to go to the movies to have fun , Wasabi is a good place to start .' # joy 
# load data
labels = []
texts = []
corpora_helper = CorporaHelper("corpora/multigenre.csv")
count_joy = 0
count_disgust = 0

for index, corpus in corpora_helper.get_data().iterrows():
    # only moviereviews
    if corpus[CorporaProperties.DOMAIN.value] == CorporaDomains.MOVIEREVIEW.value:
        # only joy
        # only disgust
        if corpus[CorporaProperties.EMOTION.value] == 'disgust':
            labels.append(0)
            texts.append(corpus[CorporaProperties.CLEANED_CORPUS.value])
            count_joy += 1
        elif corpus[CorporaProperties.EMOTION.value] == 'joy':
            labels.append(1)
            texts.append(corpus[CorporaProperties.CLEANED_CORPUS.value])
            count_disgust += 1
print('number of joy labels: ',count_joy)
print('number of disgust labels: ', count_disgust)

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
glove_dir = './glove.6B'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt', ), encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# Create embedding matrix
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

# Create model
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten()) #3D to 2D
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Load GloVe embedding
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

# Train and Evaluate
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')

# Test model
print("Evaluate on test data")
model.load_weights('pre_trained_glove_model.h5')
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# Plot performance
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

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



