import pandas as pd
from corpora_utils import CorporaHelper,CorporaDomains
from senticnet_utils import SenticNetHelper
from emotion_utils import Emotions, EmotionResult
from emotion_analyzer import EmotionAnalyzer
from random import random

"""
corpora_helper = CorporaHelper("multigenre.csv")

blog_corpora = corpora_helper.get_domain_data(CorporaDomains.MOVIEREVIEW.value)
print(CorporaDomains.BLOG.value)
print(blog_corpora)

"""
print('==============================')
print('Start script output')
print('==============================')
concept = 'the'
print('Test Senic Net with: ', concept)
snh = SenticNetHelper()
emotion = snh.get_emotion(concept)
print("SenticNet emotion: ",emotion)
print('==============================')
print("Mockup")
analyzer = EmotionAnalyzer('love', mockup=True)
print("Mockup emotionanalyzer: ",analyzer.get_emotion())
print("primary emotion: ",EmotionResult.get_primary_emotion(analyzer.get_emotion()))
print('==============================')
#corpus = 'This delicately observed story , deeply felt and masterfully stylized , is a triumph for its maverick director .'
#corpus = 'anger fear disgust sadness joy trust anticipation surprise'
#corpus = 'the boys cars are different'
#corpus = 'the old crazy man is the bad itself'
corpus = 'If you sometimes like to go to the movies to have fun , Wasabi is a good place to start .' # joy 
#corpus = 'All this not sleeping has a terrible way of playing with your memory.' # fear => test
#corpus = 'a love lot gun of fat a little '
#corpus = 'a little'
#corpus = 'boy different'
print("Analyze: ", corpus)
analyzer = EmotionAnalyzer(corpus, mockup=False)
emotion = analyzer.get_emotion(method='combine')
print("emotionanalyzer: ",emotion)
print("primary emotion: ",EmotionResult.get_primary_emotion(emotion))
print('details:')
analyzer.print_emotions()
print('==============================')
#print(analyzer._emotions)


# run over the whole corpora
#corpora_helper = CorporaHelper("multigenre.csv")
#for corpus in corpora_helper.get_data()
#    analyzer = EmotionAnalyzer(corpus, mockup=False)


#for corpus in corpora_helper.get_data