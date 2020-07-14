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

corpora_helper = CorporaHelper("dev_corpus.csv")
#corpora_helper = CorporaHelper("multigenre.csv")
starttime = time.time()
corpora_helper.remove_emotion('other')
#corpora_helper.remove_domain('moviereview')
corpora_helper.remove_duplicate_coprus()

# Preprocess corpora
# depends on the corpora which functions to use


corpora_helper.translate_urls()
corpora_helper.translate_emoticons()
corpora_helper.translate_emojis()
corpora_helper.translate_email()
#corpora_helper.translate_mention()
corpora_helper.translate_html_tags()
corpora_helper.translate_camel_case()
corpora_helper.translate_underscore()

corpora_helper.translate_string('-LRB-','(')
corpora_helper.translate_string('-RRB-',')')
corpora_helper.translate_string('`',"'") # ` to '
corpora_helper.translate_string("''",'"') # double '' to "
corpora_helper.translate_contractions()
corpora_helper.translate_string("'","") # remove '
corpora_helper.translate_string("\\n"," ") # replace new lines with space

#corpora_helper.spell_correction() # not accurate

corpora_helper.add_space_at_special_chars()
corpora_helper.add_space_at_special_chars(regexlist = r"([#])")
#corpora_helper.translate_to_lower()


print(corpora_helper.get_data())

# abbr
# slang
# custom replace

counter = 1
debug = False
    
# write result to file
corpora_helper.write_to_csv('dev_corpus_cleaned.csv')

# Time stats
endtime = time.time()
duration = endtime - starttime
print("Number of copora: ", corpora_helper.get_data().shape[0])
print("Duration: ",duration,"[s]", "-",duration/60,"[min]")

#corpora_helper = CorporaHelper()
#corpora_helper.load_corpora_from_csv(file="test.csv",sep=',')
#corpora_helper.evaluate_accurancy("test_eval_total.csv")