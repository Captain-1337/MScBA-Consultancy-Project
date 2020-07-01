from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from corpora_utils import CorporaHelper,CorporaDomains, CorporaProperties
import numpy as np
import os
import pickle
"""
Split data in Train and Test set
"""

# load data
labels = []
texts = []
corpora_helper = CorporaHelper("multigenre.csv")
# enrich some emotions with random copies to raise the number of samples
# anger from 240 to 400 => add 160 random
corpora_helper.random_enrich_emotion('anger', 160)
# fear from 263 to 400 => add 137 random
corpora_helper.random_enrich_emotion('fear', 137)
count_joy = 0
count_sadness = 0
count_anger = 0
count_fear = 0
max_per_emotion = 400

# preprocessing corpora
corpora_helper.translate_contractions() # problem space before '
corpora_helper.translate_urls() # http;/sdasd  => URL
corpora_helper.translate_camel_case()
corpora_helper.translate_underscore()
corpora_helper.add_space_at_special_chars()

for index, corpus in corpora_helper.get_data().iterrows():
    # only moviereviews
    if True or corpus[CorporaProperties.DOMAIN.value] == CorporaDomains.MOVIEREVIEW.value:
        # only anger/ fear / joy / sadness
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
max_data = count_anger + count_fear + count_joy + count_sadness
# 0 anger
# 1 fear
# 2 joy
# 3 sadness
#maxlen = 100 # max. number of words in sequences

## Create one hot encoding
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print ('%s eindeutige Tokens gefunden.' % len(word_index))

# Read the glove embedding acc. 6.1.3
"""
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

"""
# Load prepared Multigenre ensemble embedding
word_embeddings_path = 'multigenre_embedding_final_new.pkl'
with open(word_embeddings_path, 'rb') as word_embeddings_file:
    embedding_info = pickle.load(word_embeddings_file)
maxlen = embedding_info[1]

# Prepare data
#maxlen = 100 # max. number of words in sequences
#split 90% for train and 10% for test
training_samples = int(max_data * 0.9)
validation_samples = int(max_data * 0) 
test_samples = int(max_data * 0.1) 

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

# Save x,y train and test 8 2
with open('multigenre_train_test/multigenre_400_train_corpus.pkl', 'wb') as writer:
    pickle.dump(x_train, writer)
with open('multigenre_train_test/multigenre_400_train_labels.pkl', 'wb') as writer:
    pickle.dump(y_train, writer)
with open('multigenre_train_test/multigenre_400_test_corpus.pkl', 'wb') as writer:
    pickle.dump(x_test, writer)
with open('multigenre_train_test/multigenre_400_test_labels.pkl', 'wb') as writer:
    pickle.dump(y_test, writer)
# Save word_index
with open('multigenre_train_test/multigenre_400_word_index.pkl', 'wb') as writer:
    pickle.dump(word_index, writer)
