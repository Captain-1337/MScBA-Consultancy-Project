from senticnet_utils import SenticNetHelper
from emotion_lexicon import EmotionLexicon
from emotion_utils import Emotions, EmotionResult
from corpora_utils import CorporaHelper
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk import pos_tag
from itertools import combinations, permutations

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
    _correction = None

    def __init__(self, corpus = '', lexicon = None, mockup = False):
        self._corpus = corpus
        self._emotions = []
        self._mockup = mockup
        # Default Senticnet Lexicon
        if lexicon is not None:
            self._lexicon = lexicon

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
                        # get a negatation flag
                        negatation_flag = CorporaHelper.is_negated(clause)
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
                            emotions = self._lookup_combinations(word_pos,abs(x))
                            clause_emotions.extend(emotions)
                            # Permutations
                            emotions = self._lookup_permutations(word_pos,abs(x))
                            clause_emotions.extend(emotions)
                            # Lemmatized combinations
                            emotions = self._lookup_combinations(word_pos,abs(x),lemma=True)
                            clause_emotions.extend(emotions)
                            # Lemmatized permutations
                            emotions = self._lookup_permutations(word_pos,abs(x),lemma=True)
                            clause_emotions.extend(emotions)
                        
                        # lookup for single word only if not a stopword           
                        for word in self._word_pos:
                            if  not CorporaHelper.is_stopword(word["word"]):
                                
                                emotion = self._lexicon.get_emotion(word["word"],word)
                                if EmotionResult.is_neutral_emotion(emotion.get_emotion()):
                                    # try lemmatized
                                    emotion = self._lexicon.get_emotion(CorporaHelper.lemmatize_token(word["word"],word["pos"]))
                                if not EmotionResult.is_neutral_emotion(emotion.get_emotion()):
                                    clause_emotions.append(emotion)
                            # remove word from word_tokens -> cannot remove of a looping list
                            #self._word_pos.remove(word)

                        # check for instensity TODO
                        # check for Negatation
                        if negatation_flag:
                            # TODO negate clause emotions.
                            pass

                        # Add clause to total emotion            
                        self._emotions.extend(clause_emotions)                    
                
                
            elif method == 'sematic':
                # TODO implement concept/phrase extraction
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
        Reduces the ordered pos tags by 

        :param ordered_pos_tags: the pos tag list to be reduced
        :param pos_tags: the pos tags to reduce
        :returns: reduced pos tag list
        """
        ordered_pos_tags = {}
        for tag in reduce_pos_tags:
            if tag in ordered_pos_tags:
                ordered_pos_tags.popitem(tag)

        return ordered_pos_tags

    def _lookup_combinations(self, ordered_pos_tags,r,lemma=False):
        """
        Lookup combination in the lexicon

        :param ordered_pos_tags:
        :param r: number of words to combine range
        :param lemma: for lemmatized lookup
        :returns: List of emotionResults or None
        """
        emotions = []
        # TODO remove _word_ps or chacne ordered_pos_tag
        combs = combinations(ordered_pos_tags, r=r)
        
        for comb in combs:
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
                    for i in range(0,r):
                        self._word_pos.remove(comb[i])
  
                    # emotion found return
                    emotions.append(emotion)
        if emotions.count == 0:
            return None
        else:
            return emotions

    def _lookup_permutations(self, ordered_pos_tags,r,lemma=False):
        """
        Lookup permutations in the lexicon

        :param ordered_pos_tags:
        :param r: number of words for permutation
        :param lemma: for lemmatized lookup
        :returns: List of emotionResults or None
        """
        emotions = []
        # TODO remove _word_ps or chacne ordered_pos_tag
        combs = permutations(ordered_pos_tags, r=r)
        for comb in combs:
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
                    for i in range(0,r):
                        self._word_pos.remove(comb[i])
    
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
move to emotion_analyzer.py when ready
"""
"""
    def intensity_capital(self):
        # TODO VERY => intensity increased compared to very
        None

    def intensity_punctuation(self):
        # TODO !!! => intensity increased compared to single !
        None

"""