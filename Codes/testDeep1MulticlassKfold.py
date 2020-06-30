from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Conv1D
from keras import layers
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold, StratifiedKFold
from corpora_utils import CorporaHelper,CorporaDomains, CorporaProperties
import numpy as np
import os
import pickle
"""
Binary simple NN with two layers and 4 emotions
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
#number_of_classes = 4
max_per_emotion = 400

# K-Fold variables
num_folds = 10
fold_runs = 3
fold_no = 1
# trian
epochs = 10
skfold = StratifiedKFold(n_splits = num_folds, random_state = 7, shuffle = True)
acc_per_fold = []
loss_per_fold = []
avg_acc_per_run = []
avg_loss_per_run = []
create_final_model = True

# preprocessing corpora
corpora_helper.translate_contractions() # problem space before '
corpora_helper.translate_urls() # http;/sdasd  => URL
#corpora_helper.translate_emoticons()
#corpora_helper.translate_emojis()
#corpora_helper.translate_html_tags()
corpora_helper.translate_camel_case()
corpora_helper.translate_underscore()
# todo remove @blabla
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

#word_indicies_path = 'word_indices.pickle'
#with open(word_indicies_path, 'rb') as word_indicies_file:
#    word_indices = pickle.load(word_indicies_file)


#helper functions
def is_active_vector_method(string):
    return int(string)
    
def get_unigram_embedding(word, word_embedding_dict, bin_string):
    
    if word in word_embedding_dict:
        word_feature_embedding_dict = word_embedding_dict[word]
        final_embedding = np.array([])
    else:
        return None
    
    for i in range(16):
        if is_active_vector_method(bin_string[i]):
            final_embedding = np.append(final_embedding, word_feature_embedding_dict[i])
    
    return final_embedding

# 
unigram_feature_string = "1111111111111111"
#unigram_feature_string = "0000000000000000"
#word_indices_len = len(word_indices)
pre_padding = 0
embeddings_index = embedding_info[0]
MAX_SEQUENCE_LENGTH = embedding_info[1]
maxlen = MAX_SEQUENCE_LENGTH
#MAX_NB_WORDS = 10000

EMBEDDING_DIM = len(get_unigram_embedding("glad", embedding_info[0], unigram_feature_string))
print("Embedding dimension:",EMBEDDING_DIM)

# Matrix
word_embedding_matrix = list()
word_embedding_matrix = np.zeros((max_words, EMBEDDING_DIM)) # evtl. change to word_indices_len
#word_embedding_matrix.append(np.zeros(EMBEDDING_DIM))

for word, i in word_index.items(): # sorted(word_indices, key=word_indices.get):
    embedding_features = get_unigram_embedding(word, embedding_info[0], unigram_feature_string)
    if i < max_words:
        if embedding_features is not None:
            # Words not found in embedding index will be all-zeros.
            word_embedding_matrix[i] = embedding_features

word_embedding_matrix = np.asarray(word_embedding_matrix, dtype='f')
word_embedding_matrix = scale(word_embedding_matrix)

#print('word_indices_len',word_indices_len)
print('EMBEDDING_DIM',EMBEDDING_DIM)
print('input_length', MAX_SEQUENCE_LENGTH + pre_padding)
embedding = Embedding(max_words, EMBEDDING_DIM, input_length=maxlen, trainable=False)
#embedding = Embedding(word_indices_len + 1, EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH + pre_padding, trainable=False)


# Prepare data
#maxlen = 100 # max. number of words in sequences

training_samples = int(max_data * 0.8) 
validation_samples = int(max_data * 0) 
test_samples = int(max_data * 0.2) 

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

# Store x,y train and test 8 2
with open('multigenre_400_train_corpus.pkl', 'wb') as writer:
    pickle.dump(x_train, writer)
with open('multigenre_400_train_labels.pkl', 'wb') as writer:
    pickle.dump(y_train, writer)
with open('multigenre_400_test_corpus.pkl', 'wb') as writer:
    pickle.dump(x_test, writer)
with open('multigenre_400_test_labels.pkl', 'wb') as writer:
    pickle.dump(y_test, writer)


# Load x,y test
with open('multigenre_400_train_corpus.pkl', 'rb') as loader:
    x_train = pickle.load(loader)
with open('multigenre_400_train_labels.pkl', 'rb') as loader:
    y_train = pickle.load(loader)
with open('multigenre_400_test_corpus.pkl', 'rb') as loader:
    x_test = pickle.load(loader)
with open('multigenre_400_test_labels.pkl', 'rb') as loader:
    y_test = pickle.load(loader)

def create_model():
    # Create model
    """
    model = Sequential()
    model.add(embedding)
    model.add(Conv1D(32,5, activation='relu'))
    model.add(Flatten()) #3D to 2D
    #model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    #model.summary()
"""
    model = Sequential()
    model.add(embedding)
    model.add(Conv1D(32,5, activation='relu'))
    model.add(layers.Bidirectional(layers.LSTM(32,dropout=0.4, recurrent_dropout=0.4,)))
    #model.add(Dense(8, activation='relu'))
    #model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    return model
# run x Times
for run_num in range(1,fold_runs+1):
    # k-fold
    for train_ind, val_ind in skfold.split(x_train,y_train):

        # Create model
        model = create_model()

        # Load GloVe embedding
        model.layers[0].set_weights([word_embedding_matrix])
        model.layers[0].trainable = False

        # Train and Evaluate
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['acc'])
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ind run {run_num} ...')

        history = model.fit(x_train[train_ind], y_train[train_ind],
                            epochs=epochs,
                            batch_size=32,
                            verbose=1,
                            validation_data=(x_train[val_ind], y_train[val_ind]))

        # metrics
        scores = model.evaluate(x_train[val_ind], y_train[val_ind], batch_size=128)
        #print(f'Score for fold {fold_no}: {model.metrics_name[0]} of {scores[0]}; {model.metrics_name[1]} of {scores[1]*100}%')
        print(f'Score for fold {fold_no}: ... of {scores[0]}; ... of {scores[1]*100}%')
        acc_per_fold.append(scores[1]*100)
        loss_per_fold.append(scores[0])

        fold_no += 1


    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    avg_acc_per_run.append(np.mean(acc_per_fold))
    avg_loss_per_run.append(np.mean(loss_per_fold))
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')

    # reset fold vars
    acc_per_fold = []
    loss_per_fold = []
    fold_no = 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(avg_acc_per_run)):
    print('------------------------------------------------------------------------')
    print(f'> Run {i+1} Fold averages - Loss: {avg_loss_per_run[i]} - Accuracy: {avg_acc_per_run[i]}%')
print('------------------------------------------------------------------------')
print(f'Overall average scores for all {fold_runs} runs:')

print(f'> Accuracy: {np.mean(avg_acc_per_run)} (+- {np.std(avg_acc_per_run)})')
print(f'> Loss: {np.mean(avg_loss_per_run)}')
print('------------------------------------------------------------------------')

# create final model #Todo sync with fold rund
if create_final_model:
    model = create_model()
    model.summary()

    # Load GloVe embedding
    model.layers[0].set_weights([word_embedding_matrix])
    model.layers[0].trainable = False

    # Train and Evaluate
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['acc'])
    print('------------------------------------------------------------------------')
    print('Training for final model ...')

    history = model.fit(x_train[train_ind], y_train[train_ind],
                        epochs=epochs,
                        batch_size=32,
                        verbose=1)
    model.save('model_emotion_detection.h5')   

    # Test final model
    print("Evaluate Findal Model on test data")
    results = model.evaluate(x_test, y_test, batch_size=128)
    print("test loss, test acc:", results)


    # Plot performance
    """"
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
"""



