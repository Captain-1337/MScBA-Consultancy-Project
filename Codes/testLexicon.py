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
concept = 'love_you'
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
corpus = 'love'
print("Analyze: ", corpus)
analyzer = EmotionAnalyzer(corpus, mockup=False)
print("emotionanalyzer: ",analyzer.get_emotion())
print("primary emotion: ",EmotionResult.get_primary_emotion(analyzer.get_emotion()))
print('==============================')