import string, time
from itertools import combinations, permutations
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk import pos_tag
from corpora_utils import CorporaHelper,CorporaDomains,CorporaProperties
import nltk
words = []
import pandas as pd
from corpora_utils import CorporaHelper,CorporaDomains
from senticnet_utils import SenticNetHelper
from emotion_utils import Emotions, EmotionResult
from emotion_analyzer import EmotionAnalyzer
from random import random

#print(corpora_helper.get_data())

# run over the whole corpora

corpora_helper = CorporaHelper("dev_corpus_analyze.csv")
#corpora_helper = CorporaHelper("multigenre.csv")
starttime = time.time()

# Preprocess corpora
# depends on the corpora which functions to use

corpora_helper.translate_contractions()
corpora_helper.translate_urls() 
corpora_helper.translate_emoticons()
corpora_helper.translate_emojis()
corpora_helper.translate_html_tags()
corpora_helper.translate_camel_case()
corpora_helper.translate_contractions() # problem space before '
corpora_helper.translate_underscore()
corpora_helper.add_space_at_special_chars()
#corpora_helper.spell_correction()

# abbr
# slang
# custom replace

analyzer = EmotionAnalyzer()
method = 'combine'

debug = False
counter = 1

for index, corpus in corpora_helper.get_data().iterrows():
    # Analyse
    #analyzer = EmotionAnalyzer()
    emotion = analyzer.get_emotion(corpus[CorporaProperties.CLEANED_CORPUS.value], method=method)
    # Extract primary emotiom
    primary_emotion = EmotionResult.get_primary_emotion(emotion)
    # Set Result
    corpora_helper.set_calc_emotion(index, primary_emotion)
    corpora_helper.set_calc_emotion_details(index, analyzer.get_emotion_results_as_string())
    corpora_helper.set_calc_emotion_result(index, str(emotion))
    # Reset the analyzer
    
    # For debuging
    if debug :
        # print
        print("emotionanalyzer: ",str(emotion))
        print("primary emotion: ",primary_emotion)
        print('details:')
        print(analyzer.get_emotion_results_as_string())
        
        # stop after 2 
        if counter == 2:
            break
    print(counter)
    counter = counter + 1

    analyzer.reset()

    
# write result to file
corpora_helper.write_to_csv('dev_corpus_analyze_result.csv')

# Evaluate result
#corpora_helper.evaluate_accurancy("test_eval_mg_mv_combine")
# Time stats
endtime = time.time()
duration = endtime - starttime
print("Number of copora: ", corpora_helper.get_data().shape[0])
print("Duration: ",duration,"[s]", "-",duration/60,"[min]")
