from senticnet_utils import SenticNetHelper
from emotion_utils import Emotions, EmotionResult

class EmotionAnalyzer:
    """
    Class for analyzing a text corpus for having the Plutchniks basic emotions
    """
    _corpus = ''
    _emotions = []
    _mockup = False
    _emotion = ''
    _snh = SenticNetHelper()

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
        if corpus is not None:
            self._corpus = corpus
        
        # get emotion of each concept or word in the corpus 
        if self._mockup:
            # Creating some random mockup emotions for testing
            for x in range(20):
                self._emotions.append(EmotionResult.create_random_emotion())
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
                    self._emotions.append(self._snh.get_emotion(token))

            elif method == 'roger1':
                pass
            elif method == 'himmet1':
                pass


        # Calculate result emotion
        emotion = self._summarize_emotions()

        return emotion
    
    def _summarize_emotions(self):
        """
        Summarize thes emotions in one EmotionResult vector
        """
        result_emotion = EmotionResult.create_emotion()
        emotion_count = len(self._emotions)
        # Sum up the emotions
        
        for emotion in self._emotions:
            result_emotion[Emotions.ANGER.value] += emotion[Emotions.ANGER.value]
            result_emotion[Emotions.FEAR.value] += emotion[Emotions.FEAR.value]
            result_emotion[Emotions.SADNESS.value] += emotion[Emotions.SADNESS.value]
            result_emotion[Emotions.DISGUST.value] += emotion[Emotions.DISGUST.value]
            result_emotion[Emotions.JOY.value] += emotion[Emotions.JOY.value]
            result_emotion[Emotions.TRUST.value] += emotion[Emotions.TRUST.value]
            result_emotion[Emotions.SURPRISE.value] += emotion[Emotions.SURPRISE.value]
            result_emotion[Emotions.ANTICIPATION.value] += emotion[Emotions.ANTICIPATION.value]
        
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





# TODO neganation of words
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