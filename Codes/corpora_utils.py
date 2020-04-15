import pandas as pd
from enum import Enum
import unicodedata
import re
from contractions import CONTRACTION_MAP
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
""" spaCy is mostly used for lemmatizing purposes, according to the WWW it is supperior to NLTK in this matter """
import spacy                    # If you have problems installing spaCy: 
import en_core_web_sm           # Try creating a new environment in Python and do a clean spaCy install on there
nlp = en_core_web_sm.load()     # pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz



class CorporaHelper():
    """
        This class is providing some functions for corpara management....
    """

    # Datamatrix of the annodated corpora
    _data = pd.DataFrame()

    def __init__(self, file):
        self.load_corpora_from_csv(file)
        self._data

    def load_corpora_from_csv(self,file, sep=';'):
        """
            Load a corpora from a csv file into the helper 
            file: filename include the path
            sep: separator of the fields in the csv
        """
        self._data = pd.read_table(file, sep=sep)
        # TODO transform in Data structure
        """
            column 1: id
            column 2: corpus 
            column 3: emotion
            column 4: domain
        """        
        #self._data.join('cleanedcorpus')
        # column 5: cleanedcorpus
        # copy text colum into cleanedcorpus
        self._data[CorporaProperties.CLEANED_CORPUS.value] = self._data[CorporaProperties.CORPUS.value]

    def get_data(self):
        return self._data

    def get_domain_data(self, domain):

        return self._data[self._data[CorporaProperties.DOMAIN.value].str.match(domain)]

    def translate_emoticons(self):
        # TODO :-) => happy
        self._data[CorporaProperties.CLEANED_CORPUS.value]

    def translate_emojis(self):
        # TODO 😀 => happy
        None

    def remove_accent(text):
        """removes the signs on accented characters: Áccěntěd => Accented) """
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def translate_slang_words(self):
        # TODO  luv => love ...
        None

    def lemmatizer(text):
        """ Lemmatizes (root word) words with respect to verbs in capital letter: keep on keeping on! Death Stranding =>  keep on keep o ! death stranding """
        text = nlp(text)
        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
        return text

    def translate_abrevations(self):
        # TODO  => abr. = abrevation
        None

    def remove_special_char(text, remove_digits=False):
        """removes or replaces special characters: Well this was fun! 123#@! => Well this was fun 123) """
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        # text = re.sub(pattern,' ', text) # replace the special char with space
        text = re.sub(pattern, '', text)
        return text

    def translate_contractions(text, contraction_mapping=CONTRACTION_MAP):
        """expands contractions: I'm => I am   He's => He is (see contractions.py for full Map) """ 
        
            contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                        flags=re.IGNORECASE|re.DOTALL)
        
            def expand_match(contraction):
                match = contraction.group(0)
                first_char = match[0]
                expanded_contraction = contraction_mapping.get(match)\
                                    if contraction_mapping.get(match)\
                                    else contraction_mapping.get(match.lower())
                expanded_contraction = first_char+expanded_contraction[1:]
                return expanded_contraction
            expanded_text = contractions_pattern.sub(expand_match, text)
            expanded_text = re.sub("'", "", expanded_text)
            return expanded_text

    def remove_stopwords(text, is_lower_case=False):    


class CorporaDomains(Enum):
    """
        Enumarations of corpora domains
    """
    BLOG = 'blog'
    MOVIEREVIEW = 'moviereview'
    NEWSHEADLINES = 'newsheadline'
    GENERAL_TWITTER = 'general_twitter'

class CorporaProperties(Enum):
    """
        Enumarations of corpora properties => columns in the data frame
    """
    ID = 'id'
    CORPUS = 'corpus'
    EMOTION = 'emotion'
    DOMAIN = 'domain'
    CLEANED_CORPUS = 'cleanedcorpus'
    



