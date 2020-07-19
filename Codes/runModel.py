from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from sklearn.preprocessing import scale
from corpora_utils import CorporaHelper,CorporaDomains, CorporaProperties
from emotion_utils import Emotions
import numpy as np
import os
import pickle
from keras.models import load_model

model = load_model('models/model_emotion_detection_multigenre_twitter.h5')
tokenizer = pickle.load(open('models/tokenizer_multigenre_twitter.pkl', 'rb'))

max_words = 10000
maxlen = 100

## Example use of the model!
test_corpora = ['I hate you bastard ! Go away !', 'This is a lovely film , but it makes me sad .','That is because - damn it !']
test_corpora.append('Die bitch ! You make me angry .')
test_corpora.append('My son died .')
test_corpora.append('I am afraid because of this horror .')

def get_predicted_emotion(prediction:list):
    result = ''
    value = 0
    if len(prediction)>0:
        index = np.argmax(prediction[0])
        emotion = Emotions.get_emotion_text(index)
        result = emotion
    return result

for text in test_corpora:
    textarray = [text]
    #tokenizer = Tokenizer(num_words=max_words)
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
    print(get_predicted_emotion(pred))
    print("======================")