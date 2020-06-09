#from collections import defaultdict
from emotion_lexicon import EmotionLexicon
from emotion_utils import Emotions, EmotionResult
import csv

class EmoLexHelper(EmotionLexicon):
    """
    Class for the access to the NRC Emotion Lexicon
    """

    _emolex = {}

    def _is_emotion(self, word):
        """
        Checks if the annotation is an emotion and not a positive or negative value
        """
        if word in [Emotions.ANGER.value, Emotions.ANTICIPATION.value, Emotions.DISGUST.value, Emotions.FEAR.value, Emotions.JOY.value, Emotions.SADNESS.value, Emotions.SURPRISE.value, Emotions.TRUST.value]:
            return True
        else:
            return False

    def load_lexicon(self):
        """
        Loads the Lexcion from a file into a dictionary
        """       
        # load nrc wordlevel lexicon
        nrc_word_file = open("nrc_emolex/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
        csv_reader = csv.reader(nrc_word_file, delimiter='\t')
        previous_word = ''
        emotion = EmotionResult.create_emotion() # as EmotionResult dictionanry
        for line in csv_reader:
            # 0: word 1: emotion/sentiment 2:assosiation
            if len(line) == 0:
                continue
            
            if previous_word == line[0] or previous_word == '':
                if self._is_emotion(line[1]):
                    # only add emotion to the emotion dictionary -> no positive or negative
                    emotion[line[1]] = int(line[2])
            else:
                # word changed store to dictionary an reset emotion
                self._emolex[previous_word] = emotion
                emotion = EmotionResult.create_emotion()

            previous_word = line[0]

        # add last word
        self._emolex[previous_word] = emotion

    def __init__(self):
        self.load_lexicon()       

    def get_emotion(self, concept, word_pos_list=None):
        """
        Abtract function for an Lexicon to implement.
        Gets from a word or word concept an EmotionResult dictionary
        :param concept: word concept to be looked up in the lexicon
        :param word_pos_list: enhanced word part of speech token with properties. example: {word: "car",pos: "NN",order: 1, ne:""}
        :return EmotionResult:
        """
        # for lookup we remove the space inbeween the words
        concept = concept.replace(" ", "")

        if concept in self._emolex.keys():      
            emotion = self._emolex[concept]
        else:
            emotion = EmotionResult.create_neutral_emotion()

        emotion_result = EmotionResult(emotion,concept)
        return emotion_result