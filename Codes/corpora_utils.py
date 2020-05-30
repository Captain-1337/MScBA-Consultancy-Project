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
        This class is providing some functions for corpora management....
        implemented for english language
    """

    # Datamatrix of the annodated corpora
    _data = pd.DataFrame()

    def __init__(self, file=None, separator=';'):
        if file is not None:
            self.load_corpora_from_csv(file,sep=separator)

    def load_corpora_from_csv(self, file, sep=';'):
        """
            Load a corpora from a csv file into the helper 
            :param file: filename include the path
            :param sep: separator of the fields in the csv
        """
        self._data = pd.read_table(file, sep=sep, index_col=0)
        """
            id is set as index
            column 1: corpus 
            column 2: emotion
            column 3: domain
        """        
        # column 4: cleanedcorpus
        # copy text colum into cleanedcorpus
        if CorporaProperties.CLEANED_CORPUS.value in self._data.columns:
            # copy only if no cleaned corpus is set
            self._data[CorporaProperties.CLEANED_CORPUS.value] = self._data[CorporaProperties.CLEANED_CORPUS.value].astype(str) 
            for corpus_id, corpus in self.get_data().iterrows():
                if pd.isna(corpus[CorporaProperties.CLEANED_CORPUS.value]) or str(corpus[CorporaProperties.CLEANED_CORPUS.value]) == '' or corpus[CorporaProperties.CLEANED_CORPUS.value] is None or str(corpus[CorporaProperties.CLEANED_CORPUS.value]) == 'nan' :
                    self._data.at[corpus_id, CorporaProperties.CLEANED_CORPUS.value] = str(corpus[CorporaProperties.CORPUS.value])
        else:
            # full copy if column not exists 
            self._data[CorporaProperties.CLEANED_CORPUS.value] = self._data[CorporaProperties.CORPUS.value]

    def get_data(self):
        return self._data

    def get_domain_data(self, domain):
        return self._data[self._data[CorporaProperties.DOMAIN.value].str.match(domain)]

    def remove_duplicate_coprus(self):
        """
        Removes duplicate corpus
        """
        self._data.drop_duplicates(subset=[CorporaProperties.CORPUS.value, CorporaProperties.EMOTION.value, CorporaProperties.DOMAIN.value], keep='first', inplace=True)

    def remove_emotion(self, emotion):
        """
        Remove corpora with an specific emotion

        :param emotion: emotion to be droped
        """
        for corpus_id, corpus in self.get_data().iterrows():
            if corpus[CorporaProperties.EMOTION.value] == emotion:
                self._data.drop(corpus_id, inplace=True)
    
    def remove_domain(self, domain):
        """
        Remove corpora with an specific domain

        :param domain: domain to be droped
        """
        for corpus_id, corpus in self.get_data().iterrows():
            if corpus[CorporaProperties.DOMAIN.value] == domain:
                self._data.drop(corpus_id, inplace=True)

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

    def set_calc_emotion_details(self, corpus_id, emotion_details:str):
        # Set calculated emotion details
        self._data.at[corpus_id, CorporaProperties.EMOTION_DETAILS.value] = emotion_details

    def set_calc_emotion_result(self, corpus_id, emotion_result:str):
        # Set calculated emotion details
        self._data.at[corpus_id, CorporaProperties.EMOTION_RESULT.value] = emotion_result

    def write_to_csv(self, path_or_buf =''):
        """
        Writes the whole dataframe set into a CSV file

        :param path_of_buf: Path or filename to write the dataframes
        """
        self._data.to_csv(path_or_buf=path_or_buf, sep=';')

    
    def evaluate_accurancy(self, filename = ''):
        """
        Evaluate the Result of the emotion analyze
        """
        from pycm import ConfusionMatrix
        #actual_vector = self._data[CorporaProperties.EMOTION.value]
        actual_vector = self._data.loc[:,CorporaProperties.EMOTION.value].values
        #print(actual_vector)
        #predict_vector = self._data[CorporaProperties.CALCULATED_EMOTION.value]
        predict_vector = self._data.loc[:,CorporaProperties.CALCULATED_EMOTION.value].values
        #print(predict_vector)
        cm = ConfusionMatrix(actual_vector=actual_vector, predict_vector=predict_vector)
        #print(cm)
        cm.save_csv(filename)

    def translate_contractions(self):
        """
        Expands contractions in the whole corpora in the cleaned corpus
        """
        for corpus_id, corpus in self.get_data().iterrows():
            text = corpus[CorporaProperties.CLEANED_CORPUS.value]
            text = re.sub(r'\s\'','\'',text)
            self._data.at[corpus_id, CorporaProperties.CLEANED_CORPUS.value] = CorporaHelper.expand_contractions(text)

    def translate_emoticons(self): 
        """
        Replaces emoticons with text in the cleaned corpus
        """
        self._data[CorporaProperties.CLEANED_CORPUS.value] = self._data[CorporaProperties.CLEANED_CORPUS.value].apply(CorporaHelper.convert_emoticons)

    def translate_emojis(self):
        """
        Replaces an emoji with text in the cleaned corpus
        attention an unterscore is between word of the emoji text!
        """
        import emoji
        for corpus_id, corpus in self.get_data().iterrows():
            text = corpus[CorporaProperties.CLEANED_CORPUS.value]
            text = emoji.demojize(text,False,(" "," "))
            self._data.at[corpus_id, CorporaProperties.CLEANED_CORPUS.value] = text

    def translate_underscore(self):
        """
        Replaces underscore with space
        """
        self._data[CorporaProperties.CLEANED_CORPUS.value] = self._data[CorporaProperties.CLEANED_CORPUS.value].apply(lambda x: str(re.sub(r"_"," ", x)))
    
    def translate_camel_case(self):
        """
        Splits camelWords in the cleaned corpus  loveMe => love me 
        """
        self._data[CorporaProperties.CLEANED_CORPUS.value] = self._data[CorporaProperties.CLEANED_CORPUS.value].apply(CorporaHelper.camel_case_split)


    def add_space_at_special_chars(self):
        """
        Adding a space before and after special chars in the cleaned corpus
        """
        pat = re.compile(r"([.()!#:<>|/,;'?*%&{}\\^\"\[\]~§$])")
        for corpus_id, corpus in self.get_data().iterrows():
            text = corpus[CorporaProperties.CLEANED_CORPUS.value]
            text = pat.sub(" \\1 ", text)
            # remove multiple spaces
            text = re.sub(r'\s+', ' ',text)
            self._data.at[corpus_id, CorporaProperties.CLEANED_CORPUS.value] = text
    
    def spell_correction(self):
        """
        Correct spelling in the cleaned corpus
        use carefully this might also change correct spellings!
        """
        from textblob import TextBlob
        self._data[CorporaProperties.CLEANED_CORPUS.value] = self._data[CorporaProperties.CLEANED_CORPUS.value].apply(lambda x: str(TextBlob(x).correct()))

    def translate_urls(self):
        """
        Remove alls urls from the corpus
        """
        self._data[CorporaProperties.CLEANED_CORPUS.value] = self._data[CorporaProperties.CLEANED_CORPUS.value].apply(CorporaHelper.replace_urls)

    def translate_html_tags(self):
        """
        Remove all html tags from the corpus
        """
        self._data[CorporaProperties.CLEANED_CORPUS.value] = self._data[CorporaProperties.CLEANED_CORPUS.value].apply(CorporaHelper.remove_html_tags)

    def translate_slang_words(self):
        # TODO  luv => love ...
        pass

    @staticmethod
    def camel_case_split(text):
        """
        Splits camelWords in words starting with lower case loveMe => love me 
        """
        words = [[text[0]]] 
    
        pre_c = ''
        for c in text[1:]: 
            
            if words[-1][-1].islower() and c.isupper() and not pre_c.isupper(): 
                # add new word if prevoius char was lower and currentcar is upper case
                words.append(list(c.lower())) 
            else: 
                    words[-1].append(c) 
            # save previous char
            pre_c = c
        # join list to string
        result = ' '.join( [''.join(word) for word in words])
    
        return result

    @staticmethod
    def convert_emoticons(text):
        """
        Converts emoticons into words

        :param text: text to be converted
        :returns: converted text
        """
        from emot.emo_unicode import UNICODE_EMO, EMOTICONS
        for emot in EMOTICONS:
            text = re.sub(u'( '+emot+' )', " " + " ".join(EMOTICONS[emot].replace(",","").split()).lower() + " ", text)
        return text

    @staticmethod
    def replace_urls(text):
        """
        Replace URL with word URL

        :param text: text to be converted
        :returns: converted text
        """
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'URL', text)

    @staticmethod
    def remove_html_tags(text):
        """
        Remove htmls tags
        # needs bs4 and lxml
        
        :param text: text to be converted
        :returns: converted text
        """
        from bs4 import BeautifulSoup
        return BeautifulSoup(text, "lxml").text

    @staticmethod
    def remove_accent(text):
        """
        Removes the signs on accented characters: Áccěntěd => Accented) 

        :param text: text to be converted
        :returns: converted text
        """
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    @staticmethod
    def simple_stemmer(text):
        """ Stemmer removes the inflections of words and transforms it to its' BASE WORDS with respect to verbs in capital letter: My system is daily crashing ,but look now at daily. => y system is daili crash ,but look now at daily.  """
        ps = nltk.PorterStemmer()
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
        :returns: lemmatized token
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

    def translate_abbrevations(self):
        """
        Replaces abbrevatons with full text in the whole corpora
        """

        # Check this out http://lasid.sor.ufscar.br/expansion/static/index.html
        # TODO  => abr. = abrevation
        pass

    @staticmethod
    def remove_special_char(text, remove_digits=False):
        """removes or replaces special characters: Well this was fun! 123#@! => Well this was fun 123) """
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        # text = re.sub(pattern,' ', text) # replace the special char with space
        text = re.sub(pattern, '', text)
        return text


    @staticmethod
    def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
        """
        Expands contractions: I'm => I am | don't => do not | He's => He is (see contractions.py for full map) 
        :param contraction_mapping: Mapping dictionary
        """ 
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
        """
        Removes stopwords without touching the negates [no & not]: The, and, if are stopwords, computer is not => , , stopwords , computer not
        for english only
        :param text: Text to remove the stopwords
        :param is_lower_case: flag, if text is in lower case
        :returns: Text without stopwords
        """ 
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
    EMOTION_RESULT = 'emotionresult'
    EMOTION_DETAILS = 'emotiondetails'
