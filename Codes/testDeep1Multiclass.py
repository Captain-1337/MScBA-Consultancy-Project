from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from sklearn.preprocessing import scale
from corpora_utils import CorporaHelper,CorporaDomains, CorporaProperties
import numpy as np
import os
import pickle
"""
Binary simple NN with two layers and 2 emotions
"""

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
number_of_classes: 4
max_per_emotion = 240
max_data = 4*240

for index, corpus in corpora_helper.get_data().iterrows():
    # only moviereviews
    if True or corpus[CorporaProperties.DOMAIN.value] == CorporaDomains.MOVIEREVIEW.value:
        # only joy
        # only disgust
        if corpus[CorporaProperties.EMOTION.value] == 'anger':
            if max_per_emotion > count_anger:
                texts.append(corpus[CorporaProperties.CLEANED_CORPUS.value])
                labels.append(0)
                count_anger += 1
        elif corpus[CorporaProperties.EMOTION.value] == 'fear':
            if max_per_emotion > count_fear:
                texts.append(corpus[CorporaProperties.CLEANED_CORPUS.value])
                labels.append(1)
                count_fear += 1
        elif corpus[CorporaProperties.EMOTION.value] == 'joy':
            if max_per_emotion > count_joy:
                texts.append(corpus[CorporaProperties.CLEANED_CORPUS.value])
                labels.append(2)
                count_joy += 1
        elif corpus[CorporaProperties.EMOTION.value] == 'sadness':
            if max_per_emotion > count_sadness:
                texts.append(corpus[CorporaProperties.CLEANED_CORPUS.value])
                labels.append(3)
                count_sadness += 1
print('number of anger labels: ',count_anger)
print('number of fear labels: ', count_fear)
print('number of joy labels: ',count_joy)
print('number of sadness labels: ', count_sadness)
# 0 anger
# 1 fear
# 2 joy
# 3 sadness

## Create one hot encoding
maxlen = 100 # max. number of words in sequences
training_samples = int(max_data * 0.7) #672  70% of 960
validation_samples = int(max_data * 0.2) # 192 20% of 960
test_samples = int(max_data * 0.1) #96  10% of 960
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
word_embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            word_embedding_matrix[i] = embedding_vector
embedding = Embedding(max_words, embedding_dim, input_length=maxlen)

# Load prepared Multigenre embedding
"""
word_embeddings_path = 'multigenre_final_embeddings.pkl'
with open(word_embeddings_path, 'rb') as word_embeddings_file:
    embedding_info = pickle.load(word_embeddings_file)

word_indicies_path = 'word_indices.pickle'
with open(word_indicies_path, 'rb') as word_indicies_file:
    word_indices = pickle.load(word_indicies_file)


#helper functions
def is_active_vector_method(string):
    return int(string)
    
def get_unigram_embedding(word, word_embedding_dict, bin_string):
    
    word_feature_embedding_dict = word_embedding_dict[word]
    final_embedding = np.array([])
    
    for i in range(16):
        if is_active_vector_method(bin_string[i]):
            final_embedding = np.append(final_embedding, word_feature_embedding_dict[i])
    
    return final_embedding

# 
unigram_feature_string = "1111111111111111"
word_indices_len = len(word_indices)
pre_padding = 0
embeddings_index = embedding_info[0]
MAX_SEQUENCE_LENGTH = embedding_info[1]
MAX_NB_WORDS = 20000
EMBEDDING_DIM = len(get_unigram_embedding("glad", embedding_info[0], unigram_feature_string))
print("Embedding dimension:",EMBEDDING_DIM)

# Matrix
word_embedding_matrix = list()
word_embedding_matrix.append(np.zeros(EMBEDDING_DIM))

for word in sorted(word_indices, key=word_indices.get):
    embedding_features = get_unigram_embedding(word, embedding_info[0], unigram_feature_string)    
    word_embedding_matrix.append(embedding_features)

word_embedding_matrix = np.asarray(word_embedding_matrix, dtype='f')
word_embedding_matrix = scale(word_embedding_matrix)

embedding = Embedding(word_indices_len + 1, EMBEDDING_DIM,
                    input_length=MAX_SEQUENCE_LENGTH + pre_padding, trainable=False)
"""

# Create model
model = Sequential()
model.add(embedding)
model.add(Flatten()) #3D to 2D
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.summary()

# Load GloVe embedding
model.layers[0].set_weights([word_embedding_matrix])
model.layers[0].trainable = False

# Train and Evaluate
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
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

model.save('emotion_deep1.h5')

## Example use of the model!
test_corpora = ['I hate you bastard ! Go away !', 'This is a lovely film , but it makes me sad .','That is because - damn it !']
test_corpora.append('Die bitch ! You make me angry .')
test_corpora.append('My son died .')
test_corpora.append('I am afraid because of this horror .')
for text in test_corpora:
    textarray = [text]
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(textarray)
    sequences = tokenizer.texts_to_sequences(textarray)
    data = pad_sequences(sequences, maxlen=maxlen)
    print(text)
    pred = model.predict(data)
    print("prediction:", pred)

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



