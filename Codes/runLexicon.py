import string, time
from itertools import combinations, permutations
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk import pos_tag
from corpora_utils import CorporaHelper,CorporaDomains,CorporaProperties
import nltk
import pandas as pd
from corpora_utils import CorporaHelper,CorporaDomains
from senticnet_utils import SenticNetHelper
from emotion_utils import Emotions, EmotionResult
from emotion_analyzer import EmotionAnalyzer, EmotionAnalyzerRules
from random import random
from nrc_emolex_utils import EmoLexHelper
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np

# run over the whole corpora

#corpora_helper = CorporaHelper("corpora/test_mg_moviereview.csv")  
#corpora_helper = CorporaHelper("corpora/mg_strong_adv.csv", separator='\t')
#corpora_helper = CorporaHelper("corpora/mg_weak_adv.csv", separator='\t')
#corpora_helper = CorporaHelper("corpora/mg_negation.csv", separator='\t')
#corpora_helper = CorporaHelper("corpora/mg_mv_annot_testset.csv", separator='\t')
#corpora_helper = CorporaHelper("corpora/twitter_2000_test.csv", separator='\t')
#corpora_helper = CorporaHelper("corpora/multigenre_450_test.csv")
#corpora_helper = CorporaHelper("corpora/twitter_2000_test.csv", separator='\t')
#corpora_helper = CorporaHelper("corpora/twitter_2000_mg_450_test.csv", separator='\t')
corpora_helper = CorporaHelper("corpora/multigenre.csv")
#corpora_helper = CorporaHelper("corpora/twitter_all.csv", separator='\t')

result_file_name = 'lexicon_result'
# flag for all 8 emotions or just anger, fear, joy and sadness
all_emotions = False

starttime = time.time()
# remove domain
#corpora_helper.remove_domain('blog')
# remove emotions
corpora_helper.remove_emotion('other')
corpora_helper.remove_emotion('noemotion')

if not all_emotions:
    corpora_helper.remove_emotion('trust')
    corpora_helper.remove_emotion('disgust')
    corpora_helper.remove_emotion('anticipation')
    corpora_helper.remove_emotion('surprise')

# Preprocess corpora 
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

#corpora_helper.spell_correction()
corpora_helper.add_space_at_special_chars()
corpora_helper.add_space_at_special_chars(regexlist = r"([#])")
#corpora_helper.translate_to_lower()

# Init analyzer
# rules for combine
rules = EmotionAnalyzerRules()
rules.adverb_strong_modifier = True
rules.adverb_weak_modifier = True
rules.negation_shift = False
rules.negation_ratio = True
rules.noun_modifier = False

lexicon = EmoLexHelper()
#lexicon = DepecheMoodHelper() # not implemented
#lexicon = SenticNetHelper()
analyzer = EmotionAnalyzer('',lexicon, mockup=False, rules=rules)

#method = 'simple'
method = 'combine'

counter = 1
debug = False

for index, corpus in corpora_helper.get_data().iterrows():
    # Analyse
    #analyzer = EmotionAnalyzer()
    emotion = analyzer.get_emotion(corpus[CorporaProperties.CLEANED_CORPUS.value], method=method)
    # Extract primary emotiom
    if all_emotions:
        primary_emotion = EmotionResult.get_primary_emotion(emotion)
    else:
        primary_emotion = EmotionResult.get_primary_emotion(emotion, considered_emotions=['anger','fear','joy','sadness'])
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
    counter += 1

    analyzer.reset()
    
# write result to file
#corpora_helper.write_to_csv('test_mg_mv_combine.csv')
corpora_helper.write_to_csv(result_file_name + '.csv')

# Calculate Metrics PRecision Recall and F1
emo_gold = Emotions.translate_emotionlist_to_intlist(corpora_helper.get_emotions_goldlabel())
emo_calc = Emotions.translate_emotionlist_to_intlist(corpora_helper.get_emotions_calculated())

if all_emotions:    
    print(classification_report(emo_gold, emo_calc, [0, 1, 2, 3, 4, 5, 6, 7],['anger',  'fear', 'joy', 'sadness','trust','disgust','anticipation','surprise']))
else:    
    print(classification_report(emo_gold, emo_calc, [0, 1, 2, 3],['anger',  'fear', 'joy', 'sadness']))
precision, recall, f1, support = precision_recall_fscore_support(emo_gold, emo_calc)
mean_precision = np.mean(precision)
mean_recall = np.mean(recall)
mean_f1 = np.mean(f1)
print("Precision overall: ", mean_precision)
print("Recall overall: ", mean_recall)
print("F1 overall: ", mean_f1)
print("----------------------------")
#

# Evaluate result in the corpora helper
#corpora_helper.write_to_csv('test_mg_eval_mv_combine.csv')
corpora_helper.evaluate_accurancy(result_file_name + '_eval')
# Time stats
endtime = time.time()
duration = endtime - starttime
print("Number of copora: ", corpora_helper.get_data().shape[0])
print("Duration: ",duration,"[s]", "-",duration/60,"[min]")

#corpora_helper = CorporaHelper()
#corpora_helper.load_corpora_from_csv(file="test.csv",sep=',')
#corpora_helper.evaluate_accurancy("test_eval_total.csv")