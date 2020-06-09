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
        :returns: EmotionResult
        """
        pass

    def get_max_n_gramm(self):
        """
        Gets the maximum word of multiwords in the lexicon
        Default 1

        :returns: Max. Number of words in a lookup of the lexicon
        """
        return 1