import pandas as pd
from enum import Enum
import unicodedata
import re
from contractions import CONTRACTION_MAP
from negate import NEGATE
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import WordNetLemmatizer
# """ spaCy is mostly used for lemmatizing purposes, according to the WWW it is supperior to NLTK in this matter """
# import spacy                    # If you have problems installing spaCy: 
# import en_core_web_sm           # Try creating a new environment in Python and do a clean spaCy install on there
# nlp = en_core_web_sm.load() # <- pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz



class CorporaHelper():
    """
        This class is providing some functions for corpara management....
    """

    # Datamatrix of the annodated corpora
    _data = pd.DataFrame()

    def __init__(self, file):
        self.load_corpora_from_csv(file)
        self._data

    def load_corpora_from_csv(self, file, sep=';'):
        """
            Load a corpora from a csv file into the helper 
            file: filename include the path
            sep: separator of the fields in the csv
        """
        self._data = pd.read_table(file, sep=sep, index_col=0)
        # TODO transform in Data structure
        """
            id is set as index
            column 1: corpus 
            column 2: emotion
            column 3: domain
        """        
        #self._data.join('cleanedcorpus')
        # column 5: cleanedcorpus
        # copy text colum into cleanedcorpus
        self._data[CorporaProperties.CLEANED_CORPUS.value] = self._data[CorporaProperties.CORPUS.value]

    def get_data(self):
        return self._data

    def get_domain_data(self, domain):
        return self._data[self._data[CorporaProperties.DOMAIN.value].str.match(domain)]

    def set_calc_emotion(self, corpus_id, emotion:str):
        """
        Sets the calculated emotion in the dataframe

        :param corpus_id: id of the corpus
        :param emotion: calculated emotion to set
        """
        # Set calculated emotion
        self._data.at[corpus_id, CorporaProperties.CALCULATED_EMOTION.value] = emotion
        # Set a flag correct if the emotion is equal to the calculated emotion
        if self._data.at[corpus_id,CorporaProperties.EMOTION.value] == emotion:
            self._data.at[corpus_id, CorporaProperties.CORRECT.value] = 'true'

    def write_to_csv(self, path_or_buf =''):
        self._data.to_csv(path_or_buf=path_or_buf)


    def translate_emoticons(self):  # I found a list with various emoticons, some of them are tagged:
                                    # https://gist.github.com/endolith/157796
                                    # Let us decide which one to use.
        # TODO :-) => happy
        #self._data[CorporaProperties.CLEANED_CORPUS.value]
        None

    def translate_emojis(self): # check emoji.py, lets discuss which one to choose.
                                # there are over 600 emojis listed and labeled.
        # TODO 😀 => happy
        None

    def remove_accent(self, text):
        """removes the signs on accented characters: Áccěntěd => Accented) """
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def translate_slang_words(self):
        # TODO  luv => love ...
        None

    @staticmethod
    def simple_stemmer(text):
        """ Stemmer removes the inflections of words and transforms it to its' BASE WORDS with respect to verbs in capital letter: My system is daily crashing ,but look now at daily. => y system is daili crash ,but look now at daily.  """
        ps = nltk.porter.PorterStemmer()
        text = ' '.join([ps.stem(word) for word in text.split()])
        return text


    # def lemmatizer_spacy(text):
    #     """ Simmilar to Stemmer this Lemmatizes words to its' ROOT WORDS with respect to verbs in capital letter: keep on keeping on! Death Stranding =>  keep on keep on ! death stranding """
    #     text = nlp(text)
    #     text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    #     return text
    
    @staticmethod
    def lemmatize_token(token, pos):
        """
        Lemmatize a word token

        :param token: word token
        :param pos: Part of speech
        """
        from nltk.corpus import wordnet
        lemmatizer = WordNetLemmatizer()
        wn_pos = wordnet.NOUN
        # translate pos into wordnet pos
        if pos.startswith('J'):
            wn_pos = wordnet.ADJ
        elif pos.startswith('V'):
            wn_pos = wordnet.VERB
        elif pos.startswith('N'):
            wn_pos = wordnet.NOUN
        elif pos.startswith('R'):
            wn_pos = wordnet.ADV

        lem_token = lemmatizer.lemmatize(token, pos=wn_pos) 
        return lem_token

    def translate_abrevations(self):    # This one is really context specific. 
                                        # We need to discuss this
                                        # Check this out http://lasid.sor.ufscar.br/expansion/static/index.html
        # TODO  => abr. = abrevation
        None

    @staticmethod
    def remove_special_char(text, remove_digits=False):
        """removes or replaces special characters: Well this was fun! 123#@! => Well this was fun 123) """
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        # text = re.sub(pattern,' ', text) # replace the special char with space
        text = re.sub(pattern, '', text)
        return text

    @staticmethod
    def translate_contractions(text, contraction_mapping=CONTRACTION_MAP):
        """expands contractions: I'm => I am | don't => do not | He's => He is (see contractions.py for full map) """ 
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

    @staticmethod
    def remove_stopwords(text, is_lower_case=False):   
        """Removes stopwords without touching the negates [no & not]: The, and, if are stopwords, computer is not => , , stopwords , computer not """ 
        tokenizer = ToktokTokenizer()
        stopword_list = nltk.corpus.stopwords.words('english')
        stopword_list.remove('no')    # Removing negates from Stopwords
        stopword_list.remove('not')   # Removing negates from Stopwords
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in stopword_list]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)    
        return filtered_text

    @staticmethod   
    def is_stopword(word):
        """
        Checks if the word is a stopword

        :param word: word to check
        :returns: True or False
        """
        stopword_list = nltk.corpus.stopwords.words('english')
        return word in stopword_list

    @staticmethod
    def is_negated(text, include_nt=True) -> bool:
        """
        Determine if input contains negation words.
        Function retrieved from NLTK VADER (see negates)
        """
        text = [str(w).lower() for w in text]
        neg_words = []
        neg_words.extend(NEGATE)
        for word in neg_words:
            if word in text:
                return True
        if include_nt:
            for word in text:
                if "n't" in word:
                    return True
        '''if "least" in text:
            i = text.index("least")
            if i > 0 and text[i - 1] != "at":
                return True'''
        return False
    
    @staticmethod
    def allcap_differential(words):
        """
        Check whether just some words in the input are ALL CAPS
        :param list words: The words to inspect
        :returns: `True` if some but not all items in `words` are ALL CAPS
        Function retrieved from NLTK VADER 
        """
        is_different = False
        allcap_words = 0
        for word in words:
            if word.isupper():
                allcap_words += 1
        cap_differential = len(words) - allcap_words
        if 0 < cap_differential < len(words):
            is_different = True
        return is_different

    @staticmethod
    def build_ordered_pos(pos_tags):
        """
        Building an pos tag array with dictionaries
        example: {word: "car",pos: "NN",order: 1}

        :param pos_tags: tokens from pos_tag
        :returns: an ordered pos tag list
        """
        result = []
        index = 1
        for tag in pos_tags:
            pos = {}
            pos["word"] = tag[0]
            pos["pos"] = tag[1]
            pos["order"] = index
            index += 1 
            result.append(pos)
        return result


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
    CALCULATED_EMOTION = 'calcemotion'
    CORRECT = 'correct'
    



