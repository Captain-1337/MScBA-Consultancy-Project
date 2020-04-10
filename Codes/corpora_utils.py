import pandas as pd
from enum import Enum

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

    def translate_slang_words(self):
        # TODO  luv => love ...
        None

    def translate_abrevations(self):
        # TODO  => abr. = abrevation
        None

    def translate_contractions(self):
        # TODO   I'm => I am   He's => He is    John's => John  don't .... 
        # @see nltk contractions pos
        None
    



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
    



