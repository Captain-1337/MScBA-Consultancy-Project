from senticnet_utils import SenticNetHelper
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
    _snh = SenticNetHelper()
    _word_pos = []

    def __init__(self, corpus = '', mockup = False):
        self._corpus = corpus
        self._emotions = []
        self._mockup = mockup

    def get_emotion(self, corpus=None, method ='simple'):
        """
        Gets a EmotionResult frorm the corpora
        """ 
        # init
        emotion = EmotionResult.create_emotion()
        self._emotions.clear
        if corpus is not None:
            self._corpus = corpus
        
        # get emotion of each concept or word in the corpus 
        if self._mockup:
            # Creating some random mockup emotions for testing
            for x in range(20):
                self._emotions.append(EmotionResult(EmotionResult.create_random_emotion(),"mockup"))
        else:
            # TODO implement concept/phrase extraction
            # 1. Split in sentences
            # 2. Ex
            # 
            if method == 'simple':
                """
                    Simple lookup for each word token in the lexicon
                """
                from nltk.tokenize import word_tokenize
                tokens = word_tokenize(self._corpus)

                for token in tokens:
                    emotion = self._snh.get_emotion(token)
                    if not EmotionResult.is_neutral_emotion(emotion):
                        self._emotions.append(EmotionResult(emotion,token))

            elif method == 'combine':
                """
                More complex lookup for combinations and perutations of words
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
                        # Add order of word
                        word_pos = CorporaHelper.build_ordered_pos(pos_tags)
                        self._word_pos = word_pos # TODO fix
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
                        for x in range(-4, -1):
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
                        
                        for word in word_pos:
                            if  not CorporaHelper.is_stopword(word):
                                emotion = self._snh.get_emotion(word["word"])
                                if EmotionResult.is_neutral_emotion(emotion):
                                    # try lemmatized
                                    emotion = self._snh.get_emotion(CorporaHelper.lemmatize_token(word["word"],word["pos"]))
                                if not EmotionResult.is_neutral_emotion(emotion):
                                    clause_emotions.append(EmotionResult(emotion,word["word"]))
                            # remove word from word_tokens
                            self._word_pos.remove(word)

                        # check for instensity
                        # check for Negatation
                        if negatation_flag:
                            # TODO negate clause emotions.
                            pass

                    # Add clause to total emotion            
                    self._emotions.extend(clause_emotions)                    
                
                
            elif method == 'sematic':
                pass


        # Calculate result emotion
        emotion = self._summarize_emotions()

        return emotion
    
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
        :param r: number of words to combine
        :param lemma: for lemmatized lookup
        :returns: List of emotionResults or None
        """
        emotions = []
        # TODO remove _word_ps or chacne ordered_pos_tag
        combs = combinations(ordered_pos_tags, r=r)
        for comb in combs:
            # check if part of the comination not mached yet
            # check if distance is not twice as number of words
            distance = self._get_ordered_pos_tag_distance(comb)
            if distance > 2*r:
                # skip if distance is higher that twice as the number of words
                continue                    
            else:

                if lemma:
                    concept = CorporaHelper.lemmatize_token(comb[0]["word"], comb[0]["pos"]) + "_" + CorporaHelper.lemmatize_token(comb[1]["word"], comb[1]["pos"])
                    if r > 2: concept = concept + "_" + CorporaHelper.lemmatize_token(comb[2]["word"], comb[2]["pos"])
                    if r > 3: concept = concept + "_" + CorporaHelper.lemmatize_token(comb[3]["word"], comb[3]["pos"])
                else:
                    concept = comb[0]["word"] + "_" + comb[1]["word"] 
                    if r > 2: concept = concept + "_" + comb[2]["word"]
                    if r > 3: concept = concept + "_" + comb[3]["word"]

                # lookup emotion
                emotion = self._snh.get_emotion(concept)
                if not EmotionResult.is_neutral_emotion(emotion):
                    # remove tokens from word_tokens
                    self._word_pos.remove(comb[0])
                    self._word_pos.remove(comb[1])
                    if r > 2: self._word_pos.remove(comb[2])
                    if r > 3: self._word_pos.remove(comb[3])
  
                    # emotion found return
                    emotions.append(EmotionResult(emotion,concept))
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
                # check if part of the comination not mached yet
                # check if distance is not twice as number of words
                distance = self._get_ordered_pos_tag_distance(comb)
                if distance > 2*r:
                    # skip if distance is higher that twice as the number of words
                    continue                    
                else:
                    if lemma:
                        concept = CorporaHelper.lemmatize_token(comb[0]["word"], comb[0]["pos"]) + "_" + CorporaHelper.lemmatize_token(comb[1]["word"], comb[1]["pos"])
                        if r > 2: concept = concept + "_" + CorporaHelper.lemmatize_token(comb[2]["word"], comb[2]["pos"])
                        if r > 3: concept = concept + "_" + CorporaHelper.lemmatize_token(comb[3]["word"], comb[3]["pos"])
                    else:
                        concept = comb[0]["word"] + "_" + comb[1]["word"] 
                        if r > 2: concept = concept + "_" + comb[2]["word"]
                        if r > 3: concept = concept + "_" + comb[3]["word"]

                    # lookup emotion
                    emotion = self._snh.get_emotion(concept)
                    if not EmotionResult.is_neutral_emotion(emotion):
                        # remove tokens from word_tokens
                        self._word_pos.remove(comb[0])
                        self._word_pos.remove(comb[1])
                        if r > 2: self._word_pos.remove(comb[2])
                        if r > 3: self._word_pos.remove(comb[3])
    
                        # emotion found return
                        emotions.append(EmotionResult(emotion,concept))
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