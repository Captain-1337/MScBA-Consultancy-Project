import pandas as pd
from corpora_utils import CorporaHelper,CorporaDomains, CorporaProperties
from senticnet_utils import SenticNetHelper
from emotion_utils import Emotions, EmotionResult
from emotion_analyzer import EmotionAnalyzer
from random import random


#corpus = 'All this not sleeping has a terrible way of playing with your memory.' # fear => test
#corpus = "The Rock is destined to be the 21st Century s new Conan and that he s going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal ."
corpus = 'If you sometimes like to go to the movies to have fun , Wasabi is a good place to start .' # joy 
print("Analyze: ", corpus)
analyzer = EmotionAnalyzer(corpus, mockup=False)
emotion = analyzer.get_emotion(method='combine')
print("emotionanalyzer: ",emotion)
print("primary emotion: ",EmotionResult.get_primary_emotion(emotion))
print('details:')
analyzer.print_emotions()
print('==============================')
#print(analyzer._emotions)



# 