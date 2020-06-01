

class EmotionLexicon:
    """
    Class for an Emotion Lexicon
    """
    def get_emotion(self, concept, word_pos_list=None):
        """
        Abtract function for an Lexicon to implement.
        Gets from a word or word concept an EmotionResult dictionary
        :param concept: word concept to be looked up in the lexicon
        :param word_pos_list: enhanced word part of speech token with properties. example: {word: "car",pos: "NN",order: 1, ne:""}
        :return EmotionResult:
        """
        pass