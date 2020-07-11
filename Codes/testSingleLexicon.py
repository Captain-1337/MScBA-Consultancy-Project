import pandas as pd
from corpora_utils import CorporaHelper,CorporaDomains, CorporaProperties
from senticnet_utils import SenticNetHelper
from emotion_utils import Emotions, EmotionResult
from emotion_analyzer import EmotionAnalyzer, EmotionAnalyzerRules
from random import random
from nrc_emolex_utils import EmoLexHelper

#corpus = 'This delicately observed story , deeply felt and masterfully stylized , is a triumph for its maverick director .'
#corpus = 'anger fear disgust sadness joy trust anticipation surprise'
#corpus = 'the boys cars are different'
#corpus = 'the old crazy man is the bad itself'
#corpus = 'If you sometimes like to go to the movies to have fun , Wasabi is a good place to start .' # joy 
#corpus = 'All this not sleeping has a terrible way of playing with your memory.' # fear => test
#corpus = 'a love lot gun of fat a little '
#corpus = 'a little'
#===================

corpus = "Sean Penn , you love Roger Cage an apology . # love "
corpus = "I do not love this very beautiful movie"
#corpus = "Stitch is a bad mannered , ugly and destructive little **** ."
#corpus = "Hmm ."
print("Analyze: ", corpus)
#lexicon = EmoLexHelper()
#lexicon = DepecheMoodHelper()
lexicon = SenticNetHelper()

# rules for combine
rules = EmotionAnalyzerRules()
rules.adverb_strong_modifier = True
rules.adverb_weak_modifier = True
rules.negation_shift = True
rules.negation_ratio = False
rules.noun_modifier = True

analyzer = EmotionAnalyzer(corpus, lexicon, mockup=False, rules=rules)
emotion = analyzer.get_emotion(method='combine')
#emotion = analyzer.get_emotion(method='simple')
print("emotionanalyzer: ",emotion)
print("primary emotion: ",EmotionResult.get_primary_emotion(emotion))
print('details:')
print(analyzer.get_emotion_results_as_string())
print('==============================')


# improvements:
# negation
# person for you him me .... Robert ...
# amplifier
# 
# abbrevations
# slang words
#
# unit test