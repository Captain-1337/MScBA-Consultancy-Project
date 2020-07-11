from enum import Enum
from random import random

class Emotions(Enum):
    """
    Enumarations of Plutchnik's basic emotions plus noemotion
    """
    ANGER = 'anger'
    FEAR = 'fear'
    SADNESS = 'sadness'
    DISGUST = 'disgust'
    JOY = 'joy'
    TRUST = 'trust'
    ANTICIPATION = 'anticipation'
    SURPRISE = 'surprise'
    # For no Emotion
    NEUTRAL = 'noemotion'

    @staticmethod
    def get_emotion_number(emotion):
        """
        Gets a number representation of the emotion
        :param emotion: String of the emotion
        """
        if emotion == Emotions.ANGER.value:
            return 0
        elif emotion == Emotions.FEAR.value:
            return 1
        elif emotion == Emotions.JOY.value:
            return 2
        elif emotion == Emotions.SADNESS.value:
            return 3
        elif emotion == Emotions.TRUST.value:
            return 4
        elif emotion == Emotions.DISGUST.value:
            return 5
        elif emotion == Emotions.ANTICIPATION.value:
            return 6
        elif emotion == Emotions.SURPRISE.value:
            return 7
        elif emotion == Emotions.NEUTRAL.value:
            return 9
        else:
            return 9999

    @staticmethod
    def translate_emotionlist_to_intlist(emotionlist):
        """
        Translates a list of emotions into a list of coded integers

        :param emotion_number: String of the emotion
        """
        result = list()
        # translate into integers
        for e in emotionlist:
            result.append(Emotions.get_emotion_number(e))
        
        return result

        
    @staticmethod
    def get_emotion_text(emotion_number):
        """
        Gets a text representation of the emotion
        :param emotion_number: String of the emotion
        """
        if emotion_number == 0:
            return Emotions.ANGER.value
        elif emotion_number == 1:
            return Emotions.FEAR.value
        elif emotion_number == 2:
            return Emotions.JOY.value
        elif emotion_number == 3:
            return Emotions.SADNESS.value
        elif emotion_number == 4:
            return Emotions.TRUST.value
        elif emotion_number == 5:
            return Emotions.DISGUST.value
        elif emotion_number == 6:
            return Emotions.ANTICIPATION.value
        elif emotion_number == 7:
            return Emotions.SURPRISE.value
        elif emotion_number == 9:
            return Emotions.NEUTRAL.value
        else:
            return 'undefined'

class EmotionResult():
    _emotion_result = {}
    emotion_concept = ''
    _remarks = list()
    _tokens = list()

    def __init__(self, emotion = None, concept = ''):
        """
        Constructor

        :param emotion: emotion to init the result
        :param concept: Assign a concept of the emotion to the result
        """
        if emotion == None:
            emotion = EmotionResult.create_emotion()
        self._emotion_result = emotion
        self.emotion_concept = concept
        self._remarks = list()
        self._tokens = list()

    def set_concept(self, concept):
        """
        Sets a concept to the result

        :param concept: concept of the result
        """
        self.emotion_concept = concept

    def add_remark(self,remark):
        """
        Adds a remark to the result

        :param remark: String as remark
        """
        self._remarks.append(remark)

    def add_token(self,token):
        """
        Adds a "word pos order" token to the result

        :param token: word with pos and order number
        """
        self._tokens.append(token)

    def get_tokens(self):
        return self._tokens

    def is_multiword(self):
        """
        Checks if the emotion is of a multiword concept
        """
        return len(self.get_tokens()) > 1

    def is_from_adjectiv(self):
        result = False
        if not self.is_multiword() and str(self.get_tokens()[0]["pos"]).startswith('J'):
            result = True
        return result

    def get_word_order(self):
        result = 0
        if not self.is_multiword():
            result = int(self.get_tokens()[0]["order"])
        return result

    def raise_emotion_by_value(self, value:float):
        """
        Raise the main emotion by a value between 0 an 1. The max value of 1 will not be exceeded
        the other emotiosn are raised by the same ratio
        :param value: Value between 0 an 1 to raise
        """
        main_emo = EmotionResult.get_primary_emotion(self._emotion_result, noemotion_threshold = 0)
        main_emo_value = self._emotion_result.get(main_emo)
        max_raise = 1 - main_emo_value
        raise_value = min(max_raise, value)
        raise_ratio = (raise_value / main_emo_value) + 1

        
        for e in self._emotion_result:
            self._emotion_result[e] = self._emotion_result[e] * raise_ratio
        self.add_remark(f'emotion has been raised by ratio {raise_ratio}')

    def reduce_emotion_by_value(self, value:float):
        """
        Reduce the main emotion by a value between 0 an 1. The min value  will not fall below 0
        the other emotions are reduced by the same ratio

        :param value: Value between 0 an 1 to raise
        """
        main_emo = EmotionResult.get_primary_emotion(self._emotion_result, noemotion_threshold = 0)
        main_emo_value = self._emotion_result.get(main_emo)

        reduce_ratio = min(float(value / main_emo_value), 1)

        for e in self._emotion_result:
            self._emotion_result[e] = self._emotion_result[e] * (1 - reduce_ratio)
        self.add_remark(f'emotion has been reduced by ratio {reduce_ratio}')

    def reduce_emotion_by_ratio(self, ratio:float):
        """
        Reduce all emotions by a ratio between 0 an 1. 

        :param ratio: Ratio between 0 an 1 to raise e.g. 0.8 means reduce by 80%
        """
        for e in self._emotion_result:
            self._emotion_result[e] = self._emotion_result[e] * (1 - ratio)
        self.add_remark(f'emotion has been reduced by ratio {ratio}')

    def reduce_emotion_in_dimension_by_value(self, value:float):
        """
        Reduce/shift all emotions by a fixed value on the same dimension
        of the hourglass of emotions into the oposite emotions

        :param value: Value to reduce on the dimension
        """
        anger_shiftet = False
        joy_shiftet = False
        trust_shiftet = False
        anticipation_shiftet = False
        for e in self._emotion_result:
            if e == Emotions.ANGER.value and float(self._emotion_result[e]) > 0:                
                self._emotion_result[e] = self._emotion_result[e] - value
                anger_shiftet = True
                if float(self._emotion_result[e]) < 0:
                    # if < 0 then move to fear
                    self._emotion_result[Emotions.FEAR.value] = abs(self._emotion_result[e])
                    self._emotion_result[e] = 0
                    
            elif e == Emotions.FEAR.value and float(self._emotion_result[e]) > 0 and not anger_shiftet: 
                self._emotion_result[e] = self._emotion_result[e] - value
                if float(self._emotion_result[e]) < 0:
                    # if < 0 then move to anger
                    self._emotion_result[Emotions.ANGER.value] = abs(self._emotion_result[e])
                    self._emotion_result[e] = 0

            elif e == Emotions.JOY.value and float(self._emotion_result[e]) > 0: 
                self._emotion_result[e] = self._emotion_result[e] - value
                joy_shiftet = True
                if float(self._emotion_result[e]) < 0:
                    # if < 0 then move to sadness
                    self._emotion_result[Emotions.SADNESS.value] = abs(self._emotion_result[e])
                    self._emotion_result[e] = 0
            elif e == Emotions.SADNESS.value and float(self._emotion_result[e]) > 0 and not joy_shiftet: 
                self._emotion_result[e] = self._emotion_result[e] - value
                if float(self._emotion_result[e]) < 0:
                    # if < 0 then move to joy
                    self._emotion_result[Emotions.JOY.value] = abs(self._emotion_result[e])
                    self._emotion_result[e] = 0

            elif e == Emotions.TRUST.value and float(self._emotion_result[e]) > 0: 
                self._emotion_result[e] = self._emotion_result[e] - value
                trust_shiftet = True
                if float(self._emotion_result[e]) < 0:
                    # if < 0 then move to disgust
                    self._emotion_result[Emotions.DISGUST.value] = abs(self._emotion_result[e])
                    self._emotion_result[e] = 0
            elif e == Emotions.DISGUST.value and float(self._emotion_result[e]) > 0 and not trust_shiftet: 
                self._emotion_result[e] = self._emotion_result[e] - value
                if float(self._emotion_result[e]) < 0:
                    # if < 0 then move to trust
                    self._emotion_result[Emotions.TRUST.value] = abs(self._emotion_result[e])
                    self._emotion_result[e] = 0

            elif e == Emotions.ANTICIPATION.value and float(self._emotion_result[e]) > 0: 
                self._emotion_result[e] = self._emotion_result[e] - value
                anticipation_shiftet = True
                if float(self._emotion_result[e]) < 0:
                    # if < 0 then move to surprise
                    self._emotion_result[Emotions.SURPRISE.value] = abs(self._emotion_result[e])
                    self._emotion_result[e] = 0
            elif e == Emotions.SURPRISE.value and float(self._emotion_result[e]) > 0 and not anticipation_shiftet: 
                self._emotion_result[e] = self._emotion_result[e] - value
                if float(self._emotion_result[e]) < 0:
                    # if < 0 then move to anticipation
                    self._emotion_result[Emotions.ANTICIPATION.value] = abs(self._emotion_result[e])
                    self._emotion_result[e] = 0

        self.add_remark(f'emotions shifted by value {value}')


    @staticmethod
    def get_emotion_by_order(emotion_results, order):
        """
        Obtains an emotion result with the desired order id from the emotion result list

        :param emotion_results: emotion Result
        :param order: Number of the order
        :returns: Single emotion result or None
        """
        result = None
        for e in emotion_results:
            if not e.is_multiword() and e.get_tokens()[0]["order"] == order:
                result = e
        return result

    def is_from_noun(self):
        result = False
        if not self.is_multiword() and str(self.get_tokens()[0]["pos"]).startswith('N'):
            result = True
        return result

    def is_from_adverb(self):
        result = False
        if not self.is_multiword() and str(self.get_tokens()[0]["pos"]).startswith('R'):
            result = True
        return result

    def get_emotion(self):
        """
        Returns the emotion dictionary

        :returns: Return the emotion dictionary
        """
        return self._emotion_result

    def get_anger(self):
        return self._emotion_result[Emotions.ANGER.value]
    
    def get_fear(self):
        return self._emotion_result[Emotions.FEAR.value]
    
    def get_sadness(self):
        return self._emotion_result[Emotions.SADNESS.value]

    def get_disgust(self):
        return self._emotion_result[Emotions.DISGUST.value]

    def get_joy(self):
        return self._emotion_result[Emotions.JOY.value]

    def get_trust(self):
        return self._emotion_result[Emotions.TRUST.value]
    
    def get_anticipation(self):
        return self._emotion_result[Emotions.ANTICIPATION.value]

    def get_surprise(self):
        return self._emotion_result[Emotions.SURPRISE.value]
    
    def print(self):
        print(self.emotion_concept)
        print(self._emotion_result)   

    def to_string(self):
        result = self.emotion_concept + " = "
        result = result + str(self._emotion_result)+ '; ' + self.tokens_to_string() + '; ' +  self.remarks_to_string() 
        return result

    def remarks_to_string(self):
        """
        Return remarks as string
        :returns: Remarks
        """
        result = ''
        for remark in self._remarks:
            if result != '':
                result = result + '; '
            result = result + str(remark)
        return result

    def tokens_to_string(self):
        """
        Return tokens as string
        :returns: Tokendetails
        """
        result = ''
        for token in self._tokens:
            if result != '':
                result = result + '; '
            result = result + str(token)
        return result
        
    @staticmethod
    def create_emotion(anger = 0, fear = 0, sadness = 0, disgust = 0, joy = 0, trust = 0, anticipation = 0, surprise = 0):
        """
        Creates an EmotionResult vector
        """
        emotion = {
           Emotions.ANGER.value: anger,
           Emotions.FEAR.value: fear,
           Emotions.SADNESS.value: sadness,
           Emotions.DISGUST.value: disgust,
           Emotions.JOY.value: joy,
           Emotions.TRUST.value: trust,
           Emotions.ANTICIPATION.value: anticipation,  
           Emotions.SURPRISE.value: surprise,
        }
        return emotion

    @staticmethod
    def create_neutral_emotion():
        """
        Creating an "empty" emotion
        """
        emotion = EmotionResult.create_emotion()
        return emotion

    @staticmethod
    def create_random_emotion():
        """
        Generating a random emotion
        """
        if random() < 0.9:
            emotion = EmotionResult.create_emotion(anger = random(), fear = random(), sadness = random(), disgust = random(), joy = random(), trust = random(), anticipation = random(), surprise = random())
        else:
            emotion = EmotionResult.create_neutral_emotion()
        
        return emotion
    
    @staticmethod
    def get_primary_emotion(emotion, noemotion_threshold = 0.0, considered_emotions = [Emotions.ANGER.value,Emotions.FEAR.value,Emotions.SADNESS.value,Emotions.DISGUST.value,Emotions.JOY.value,Emotions.TRUST.value,Emotions.SURPRISE.value,Emotions.ANTICIPATION.value]):
        """
        Get the primary emotion out of an EmotionResult

        :param emotion: emotion as EmotionsResult data vector
        :param noemotion_threshold: Sets the minimum value of the primary emotion. Below this value it will count as neutral no emotion.
        :param considered_emotions: list of the basic emotions to be considered for the output. By default all 8 emotions are considered.
        :returns: result emotion as a string
        """
        result_emotion = ''
        emotion_intensity = 0.0

        if emotion is None: 
            return result_emotion

        # Get the emotion with the highest value
        if Emotions.ANGER.value in considered_emotions: 
            result_emotion = Emotions.ANGER.value
            emotion_intensity = emotion[Emotions.ANGER.value]
        if Emotions.FEAR.value in considered_emotions and float(emotion[Emotions.FEAR.value]) > emotion_intensity:
            result_emotion = Emotions.FEAR.value
            emotion_intensity = float(emotion[Emotions.FEAR.value])
        if Emotions.SADNESS.value in considered_emotions and float(emotion[Emotions.SADNESS.value]) > emotion_intensity:
            result_emotion = Emotions.SADNESS.value
            emotion_intensity = float(emotion[Emotions.SADNESS.value])
        if Emotions.DISGUST.value in considered_emotions and float(emotion[Emotions.DISGUST.value]) > emotion_intensity:
            result_emotion = Emotions.DISGUST.value
            emotion_intensity = float(emotion[Emotions.DISGUST.value])
        if Emotions.JOY.value in considered_emotions and float(emotion[Emotions.JOY.value]) > emotion_intensity:
            result_emotion = Emotions.JOY.value
            emotion_intensity = float(emotion[Emotions.JOY.value])
        if Emotions.TRUST.value in considered_emotions and float(emotion[Emotions.TRUST.value]) > emotion_intensity:
            result_emotion = Emotions.TRUST.value
            emotion_intensity = float(emotion[Emotions.TRUST.value])
        if Emotions.SURPRISE.value in considered_emotions and float(emotion[Emotions.SURPRISE.value]) > emotion_intensity:
            result_emotion = Emotions.SURPRISE.value
            emotion_intensity = float(emotion[Emotions.SURPRISE.value])
        if Emotions.ANTICIPATION.value in considered_emotions and float(emotion[Emotions.ANTICIPATION.value]) > emotion_intensity:
            result_emotion = Emotions.ANTICIPATION.value
            emotion_intensity = float(emotion[Emotions.ANTICIPATION.value])       
        
        # Check if emotion intesity is hight enough (> than the threshold) otherwise it is neutral
        if emotion_intensity < noemotion_threshold:
            result_emotion = Emotions.NEUTRAL.value

        return result_emotion

    @staticmethod
    def is_neutral_emotion(emotion):
        """
        Checks if the emotion is a neutral emotion (all emotions equal 0)

        :param emotion: emotion dict
        :returns boolean: True if it is a neutral emotion
        """
        result = emotion[Emotions.FEAR.value] == 0 and emotion[Emotions.ANGER.value] == 0
        result = result and emotion[Emotions.JOY.value] == 0 and emotion[Emotions.ANTICIPATION.value] == 0
        result = result and emotion[Emotions.SADNESS.value] == 0 and emotion[Emotions.DISGUST.value] == 0
        result = result and emotion[Emotions.SURPRISE.value] == 0 and emotion[Emotions.TRUST.value] == 0

        return result
    