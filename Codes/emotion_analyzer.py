from senticnet_utils import SenticNetHelper
from emotion_lexicon import EmotionLexicon
from emotion_utils import Emotions, EmotionResult
from corpora_utils import CorporaHelper
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk import pos_tag
from itertools import combinations, permutations
import gensim.downloader as api # expirimental use
from adverb_class_list import ADVERBS_STRONG_INT, ADVERBS_WEAK_INT, ADVERBS_DOUBT


class EmotionAnalyzerRules:
    """
    Class for rule enabled of the EmotionAnalyzer
    """
    adverb_strong_modifier = True
    adverb_weak_modifier = True
    negation_shift = True
    negation_ratio = False
    noun_modifier = True
    #modifier = True
    #amplifier = True
    #adversatives = True
    #firstperson = True
     


class EmotionAnalyzer:
    """
    Class for analyzing a text corpus for having the Plutchniks basic emotions
    """
    _corpus = ''
    _emotions = []
    _mockup = False
    _emotion = ''
    _lexicon = SenticNetHelper()
    _word_pos = []
    _org_word_pos = []
    _correction = None
    # properties for similarity (experimental)
    _use_similar_words_lookup = False
    _similarity_level = 0.5
    _similarity_lookup_topn = 10
    _word_vector = None
    _rules = EmotionAnalyzerRules()

    def __init__(self, corpus = '', lexicon = None, mockup = False, rules = EmotionAnalyzerRules()):
        self._corpus = corpus
        self._emotions = []
        self._mockup = mockup
        # Default Senticnet Lexicon
        if lexicon is not None:
            self._lexicon = lexicon
        
        if rules is not None:
            self._rules = rules

    def enabel_similarity_lookup(self, simalirity_level = 0.5 ,topn = 10):
        """
        Enables to use similar word lookup in word2vec embedding for not found words in lexcion
        :param simalirity_level: the min value of word similarity
        :param topn: number of max similar words to lookup
        """
        self._use_similar_words_lookup = True
        self._similarity_level = simalirity_level
        self._topn = topn
        if self._word_vector == None:
            self._word_vector = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data


    def disable_similarity_lookup(self):
        """
        Disable the similar word lookup for not found words in lexcion
        :param simalirity_level: the min value of word similarity
        :param topn: number of max similar words to lookup
        """
        self._use_similar_words_lookup = False


    def reset(self):
        """
        Resets the Analyzer
        """
        self._corpus = ''
        self._emotions = []
        self._mockup = False
        self._emotion = ''
        self._word_pos = []
        self._correction = None
        #self.disable_similarity_lookup()

    def set_emotion_correction(self, correction):
        """
        Set a correction to balance between corpora and lexicon
        :param correction: Values to correct the difference beween corpora and lexicon as EmotionResult
        """
        self._correction = correction

    def get_emotion(self, corpus=None, method ='simple'):
        """
        Gets a EmotionResult frorm the corpora
        """ 
        # init
        emotion = EmotionResult.create_emotion()
        self._emotions.clear()
        if corpus is not None:
            self._corpus = corpus
        
        # get emotion of each concept or word in the corpus 
        if self._mockup:
            # Creating some random mockup emotions for testing
            for x in range(20):
                self._emotions.append(EmotionResult(EmotionResult.create_random_emotion(),"mockup"))
        else:
            
            if method == 'simple':
                """
                    Simple lookup for each word token in the lexicon
                """
                from nltk.tokenize import word_tokenize
                tokens = word_tokenize(self._corpus)

                for token in tokens:
                    emotion = self._lexicon.get_emotion(token)
                    if not EmotionResult.is_neutral_emotion(emotion.get_emotion()):
                        self._emotions.append(emotion)

            elif method == 'combine':
                """
                More complex lookup for combinations and perutations of multi words
                """
                # Split into sentences
                sentences = sent_tokenize(self._corpus)
                # For each sentence 
                for sent in sentences:
                    # Split into clauses
                    clauses = sent.split(',')
                    for clause in clauses:
                        #  Word tokenize => list of words without punctation
                        tokenizer = RegexpTokenizer(r'\w+')                        
                        word_tokens = tokenizer.tokenize(clause)
                        pos_tags = pos_tag(word_tokens)
                        # Add order of word and named entity label
                        word_pos = CorporaHelper.build_ordered_pos(pos_tags, add_ne_label = True)
                        self._word_pos = word_pos # TODO fix no need as class variable
                        self._org_word_pos = word_pos.copy() # TODO save of the origin word pos list
                        
                        # get a negatation flag                        
                        negatation_flag = CorporaHelper.is_negated(word_tokens)
                        #print("negatation_flag",negatation_flag)

                        # search for concept phrasen through all word in the sentence
                        # combinations of 4 
                        # - permutations of 4
                        # - - lemmatize
                        # - - combination of 4
                        # - - permutaions of 4
                        # the same for 3 and 2 word combinations
                        #
                        # for 4 to 2 multiword concepts
                        clause_emotions = []

                        # get max of word lookup
                        max_lex_words = self._lexicon.get_max_n_gramm()
                        range_from = max_lex_words * -1
                        range_to = -1
                        # lookup for combinations of 2 or more words                
                        for x in range(range_from, range_to):
                            
                            # Combinations
                            combs = combinations(word_pos, r=abs(x))
                            emotions = self._lookup_word_combinations(combs, abs(x))
                            clause_emotions.extend(emotions)
                            # Permutations
                            combs = permutations(word_pos, r=abs(x))
                            emotions = self._lookup_word_combinations(combs, abs(x))                       
                            clause_emotions.extend(emotions)
                            # Lemmatized combinations
                            combs = combinations(word_pos, r=abs(x))
                            emotions = self._lookup_word_combinations(combs, abs(x), lemma=True)
                            clause_emotions.extend(emotions)
                            # Lemmatized permutations
                            combs = permutations(word_pos, r=abs(x))
                            emotions = self._lookup_word_combinations(combs, abs(x), lemma=True)
                            clause_emotions.extend(emotions)
                        
                        # lookup for single word only if not a stopword           
                        for word in self._word_pos:
                            if  not CorporaHelper.is_stopword(word["word"]):
                                
                                emotion = self._lexicon.get_emotion(word["word"],word)
                                if EmotionResult.is_neutral_emotion(emotion.get_emotion()):
                                    # no emotion found try lemmatized
                                    emotion = self._lexicon.get_emotion(CorporaHelper.lemmatize_token(word["word"],word["pos"]))

                                if not EmotionResult.is_neutral_emotion(emotion.get_emotion()):
                                    emotion.add_token(word)
                                    clause_emotions.append(emotion)
                            # remove word from word_tokens -> cannot remove of a looping list
                            #self._word_pos.remove(word)
                        
                        # Linguistic rules some semantic features

                        # Adverb - Adjektive
                        # Strong modifier
                        if self._rules.adverb_strong_modifier:
                            weight = 0.5
                            for token in self._org_word_pos:
                                if token['pos'].startswith('R'):
                                    # Adverb
                                    adv_order = token['order']
                                    # look ahead for adjective or adverb with emotion
                                    emo = EmotionResult.get_emotion_by_order(clause_emotions, adv_order + 1)
                                    if emo is not None and ( emo.is_from_adjectiv() or emo.is_from_adverb()):
                                        if token['word'] in ADVERBS_STRONG_INT:
                                            # APS Scoring
                                            adv_intensity = 1 # for all adverb the same value
                                            emo.raise_emotion_by_value(weight * adv_intensity)
                                            e = EmotionResult.get_emotion_by_order(clause_emotions, adv_order)
                                            # get emotion of the token and remove the adverb from result
                                            if e is not None:
                                                clause_emotions.remove(e)

                        # Weak modifier
                        if self._rules.adverb_weak_modifier:
                            weight = 0.5
                            for token in self._org_word_pos:
                                if token['pos'].startswith('R'):
                                    # Adverb
                                    adv_order = token['order']
                                    # look ahead for adjective or adverb with emotion
                                    emo = EmotionResult.get_emotion_by_order(clause_emotions, adv_order + 1)
                                    if emo is not None and (emo.is_from_adjectiv() or emo.is_from_adverb()):
                                        if token['word'] in ADVERBS_WEAK_INT or token['word'] in ADVERBS_DOUBT:
                                            # APS Scoring
                                            adv_intensity = 1 # for all adverb the same value
                                            emo.reduce_emotion_by_value(weight * adv_intensity)
                                            e = EmotionResult.get_emotion_by_order(clause_emotions, adv_order)
                                            # get emotion of the token and remove the adverb from result
                                            if e is not None:
                                                clause_emotions.remove(e)

                        # Adjectiv Modifier of nouns
                        # for each adjective in word pos overwrites the noun emotion (if not a multiword)
                        if self._rules.noun_modifier:
                            for e in clause_emotions:
                                if e.is_multiword():
                                    # skip multiwords keep their emotions
                                    continue
                                else:
                                    if e.is_from_adjectiv():
                                        adj_order = e.get_word_order()
                                        # look ahead
                                        emo = EmotionResult.get_emotion_by_order(clause_emotions, adj_order + 1)
                                        if emo is not None and emo.is_from_noun():
                                            # Remove the noun emotion
                                            e.add_remark(f'Noun emotion of "{emo.emotion_concept}" has been removed and was "overwritten" by this adjective')
                                            clause_emotions.remove(emo)
                        
                        # Negation handling
                        # reduce by a ration
                        if self._rules.negation_ratio and negatation_flag:
                            negation_ratio = 0.8
                            #print("negation handling 1")
                            for e in clause_emotions:
                                e.reduce_emotion_by_ratio(negation_ratio)

                        # alternative negation handling
                        # shift by a fixed value on the dimension of the hourglass of emotions
                        if self._rules.negation_shift and negatation_flag:
                            negation_shift_value = 0.8
                            #print("negation handling 2")
                            for e in clause_emotions:
                                e.reduce_emotion_in_dimension_by_value(negation_shift_value, primary_emotion_only = True)

                        #TODO Add more rules

                        # Add clause to total emotion
                        self._emotions.extend(clause_emotions)                    
                
            elif method == 'sematic':
                # TODO implement concept/phrase extraction
                # create semantic tree

                pass


        # Calculate result emotion
        emotion = self._summarize_emotions()

        # Add a correction of the emotion
        if self._correction is not None:
            emotion[Emotions.ANGER.value] = emotion[Emotions.ANGER.value] + self._correction[Emotions.ANGER.value]
            emotion[Emotions.FEAR.value] = emotion[Emotions.FEAR.value] + self._correction[Emotions.FEAR.value]
            emotion[Emotions.SADNESS.value] = emotion[Emotions.SADNESS.value] + self._correction[Emotions.SADNESS.value]
            emotion[Emotions.DISGUST.value] = emotion[Emotions.DISGUST.value] + self._correction[Emotions.DISGUST.value]
            emotion[Emotions.JOY.value] = emotion[Emotions.JOY.value] + self._correction[Emotions.JOY.value]
            emotion[Emotions.TRUST.value] = emotion[Emotions.TRUST.value] + self._correction[Emotions.TRUST.value]
            emotion[Emotions.SURPRISE.value] = emotion[Emotions.SURPRISE.value] + self._correction[Emotions.SURPRISE.value]
            emotion[Emotions.ANTICIPATION.value] = emotion[Emotions.ANTICIPATION.value] + self._correction[Emotions.ANTICIPATION.value]

        return emotion


    def get_similar_words(self, word):
        """
        Get similar word out of a word2vec embedding or wordnet 
        :param word:
        :returns: list of words
        """
        #TODO
        pass

    def get_sum_emotion_result(self):
        #TODO
        pass
    
    def get_emotion_results(self):
        #TODO
        return self._emotions

    def get_emotion_results_as_string(self):
        result = ''
        for emotion in self._emotions:
            result = result + emotion.to_string() + chr(13) + chr(10)
        return result
    
    def print_emotions(self):
        for emotion in self._emotions:
            emotion.print()
    
    def _get_ordered_pos_tag_distance(self,ordered_pos_tags):
        """
        Get the max distance of the pos tag in the clause

        :param ordered_pos_tags: ordered pos tag dictionary with key "order"
        :returns: Max. distance as integer
        """
        order_values = [] 
        for tag in ordered_pos_tags:
            order_values.append(tag["order"])
        return max(order_values)-min(order_values)

    def _reduce_ordered_pos_tags(self,ordered_pos_tags, reduce_pos_tags):
        """
        Reduces the ordered pos tags by a list of pos tags

        :param ordered_pos_tags: the pos tag list to be reduced
        :param pos_tags: the pos tags to reduce
        :returns: reduced pos tag list
        """
        ordered_pos_tags = {}
        for tag in reduce_pos_tags:
            if tag in ordered_pos_tags:
                ordered_pos_tags.popitem(tag)

        return ordered_pos_tags

    def _lookup_word_combinations(self, combinations ,r ,lemma=False):
        """
        Lookup combinations of ordered word pos tags in the lexicon

        :param combinations: combinations to lookup
        :param r: number of words to combine range
        :param lemma: for lemmatized lookup
        :returns: List of emotionResults or None
        """
        emotions = []
        
        for comb in combinations:
            skip = False

            # check if part of the comination not mached yet => skip if part of the combination has been mached yet.
            for i in range(0,r):
                if comb[i] not in self._word_pos:
                    skip = True
                    continue

            # check if distance is not twice as number of words
            distance = self._get_ordered_pos_tag_distance(comb)
            if distance > 2*r or skip:
                # skip if distance is higher that twice as the number of words
                skip = False
                continue                    
            else:
                concept = ''
                concept_pos = []

                # build concept and concept pos
                for i in range(0,r):
                    if lemma:
                        concept = concept + " " + CorporaHelper.lemmatize_token(comb[i]["word"], comb[i]["pos"])
                    else:
                        concept = concept + " " + comb[i]["word"]
                    concept_pos.append(comb[i])
                concept = concept.strip()

                # lookup emotion
                emotion = self._lexicon.get_emotion(concept,concept_pos)
                if not EmotionResult.is_neutral_emotion(emotion.get_emotion()):
                    # remove tokens from word_tokens
                    for token_pos in concept_pos:
                        emotion.add_token(token_pos)
                        self._word_pos.remove(token_pos)
  
                    # emotion found return                    
                    emotions.append(emotion)
        if emotions.count == 0:
            return None
        else:
            return emotions
 
    def _summarize_emotions(self):
        """
        Summarize thes emotions in one EmotionResult vector
        """
        result_emotion = EmotionResult.create_emotion()
        emotion_count = len(self._emotions)

        # Sum up the emotions
        for emotion in self._emotions:
            result_emotion[Emotions.ANGER.value] += emotion.get_anger()
            result_emotion[Emotions.FEAR.value] += emotion.get_fear()
            result_emotion[Emotions.SADNESS.value] += emotion.get_sadness()
            result_emotion[Emotions.DISGUST.value] += emotion.get_disgust()
            result_emotion[Emotions.JOY.value] += emotion.get_joy()
            result_emotion[Emotions.TRUST.value] += emotion.get_trust()
            result_emotion[Emotions.SURPRISE.value] += emotion.get_surprise()
            result_emotion[Emotions.ANTICIPATION.value] += emotion.get_anticipation()
        
        # Normalize
        if emotion_count > 0:
            result_emotion[Emotions.ANGER.value] = result_emotion[Emotions.ANGER.value]/emotion_count
            result_emotion[Emotions.FEAR.value] = result_emotion[Emotions.FEAR.value]/emotion_count
            result_emotion[Emotions.SADNESS.value] = result_emotion[Emotions.SADNESS.value]/emotion_count
            result_emotion[Emotions.DISGUST.value] = result_emotion[Emotions.DISGUST.value]/emotion_count
            result_emotion[Emotions.JOY.value] = result_emotion[Emotions.JOY.value]/emotion_count
            result_emotion[Emotions.TRUST.value] = result_emotion[Emotions.TRUST.value]/emotion_count
            result_emotion[Emotions.SURPRISE.value] = result_emotion[Emotions.SURPRISE.value]/emotion_count
            result_emotion[Emotions.ANTICIPATION.value] = result_emotion[Emotions.ANTICIPATION.value]/emotion_count

        return result_emotion

    def _lemmatize_token(self, pos_token):
        """
        Lemmatize a pos token and returns the lemmatized word
        :param pos_token: Token with part of speech
        :returns: Returns the lemmatized token
        """
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import wordnet

        lemmatizer = WordNetLemmatizer()
        # translate ot wordnet pos
        pos = pos_token[1]
        wn_pos = ''
        if pos.startswith('J'):
            wn_pos = wordnet.ADJ
        elif pos.startswith('V'):
            wn_pos = wordnet.VERB
        elif pos.startswith('N'):
            wn_pos = wordnet.NOUN
        elif pos.startswith('R'):
            wn_pos = wordnet.ADV
        else:        
            wn_pos = ''

        word = lemmatizer.lemmatize(pos_token[0], pos=wn_pos)

        return word


# TODO negataion of words
# TODO Check intesitiy happy vs. HAPPY
"""
    def intensity_capital(self):
        # TODO VERY => intensity increased compared to very
        None

    def intensity_punctuation(self):
        # TODO !!! => intensity increased compared to single !
        None

"""