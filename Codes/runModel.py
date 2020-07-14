from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from sklearn.preprocessing import scale
from corpora_utils import CorporaHelper,CorporaDomains, CorporaProperties
import numpy as np
import os
import pickle
from keras.models import load_model

model = load_model('model_emotion_detection.h5')
max_words = 10000
maxlen = 100

## Example use of the model!
test_corpora = ['I hate you bastard ! Go away !', 'This is a lovely film , but it makes me sad .','That is because - damn it !']
test_corpora.append('Die bitch ! You make me angry .')
test_corpora.append('other than that hed get offended or pissed off or ignorant about it .')
test_corpora.append('wash your damned hands .')
for text in test_corpora:
    textarray = [text]
    tokenizer = Tokenizer(num_words=max_words, filters='')
    tokenizer.fit_on_texts(textarray)
    sequences = tokenizer.texts_to_sequences(textarray)
    data = pad_sequences(sequences, maxlen=maxlen)
    print("Text: ", text)
    pred = model.predict(data)
    print("prediction:", pred)
    print("anger:", pred[0][0])
    print("fear:", pred[0][1])
    print("joy:", pred[0][2])
    print("sadness:", pred[0][3])
    print("======================")