from enum import Enum
from random import random

class EmotionResult():
    _emotion_result = {}
    emotion_context = ''

    def __init__(self, emotion_result, context = ''):
        """
        Constructor

        :param emotion_result: inti the Instace with that emotion_result dictionary
        :param context: Assign a context of the emotion to the result
        """
        if emotion_result == None:
            emotion_result = EmotionResult.create_emotion()
        self._emotion_result = emotion_result
        self.emotion_context = context

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
        print(self.emotion_context)
        print(self._emotion_result)        
        
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
           Emotions.TRUST.value:trust,
           Emotions.ANTICIPATION.value:anticipation,  
           Emotions.SURPRISE.value:surprise,
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
    def get_primary_emotion(emotion, neutral_threshold = 0.1):
        """
        Get the primary emotion out of an EmotionResult

        :param emotion: emotion as EmotionsResult data vector
        :param neutral_threshold: Sets the minimum value of the primary emotion. Below this value it will count as neutral no emotion.
        """
        result_emotion =''
        emotion_intensity = 0.0

        if emotion is None: 
            return result_emotion

        # Get the emotion with the highest value
        result_emotion = Emotions.ANGER.value
        emotion_intensity = emotion[Emotions.ANGER.value]
        if float(emotion[Emotions.FEAR.value]) > emotion_intensity:
            result_emotion = Emotions.FEAR.value
            emotion_intensity = float(emotion[Emotions.FEAR.value])
        if float(emotion[Emotions.SADNESS.value]) > emotion_intensity:
            result_emotion = Emotions.SADNESS.value
            emotion_intensity = float(emotion[Emotions.SADNESS.value])
        if float(emotion[Emotions.DISGUST.value]) > emotion_intensity:
            result_emotion = Emotions.DISGUST.value
            emotion_intensity = float(emotion[Emotions.DISGUST.value])
        if float(emotion[Emotions.JOY.value]) > emotion_intensity:
            result_emotion = Emotions.JOY.value
            emotion_intensity = float(emotion[Emotions.JOY.value])
        if float(emotion[Emotions.TRUST.value]) > emotion_intensity:
            result_emotion = Emotions.TRUST.value
            emotion_intensity = float(emotion[Emotions.TRUST.value])
        if float(emotion[Emotions.SURPRISE.value]) > emotion_intensity:
            result_emotion = Emotions.SURPRISE.value
            emotion_intensity = float(emotion[Emotions.SURPRISE.value])
        if float(emotion[Emotions.ANTICIPATION.value]) > emotion_intensity:
            result_emotion = Emotions.ANTICIPATION.value
            emotion_intensity = float(emotion[Emotions.ANTICIPATION.value])       
        
        # Check if emotion intesity is hight enough (> than the threshold) otherwise it is neutral
        if emotion_intensity < neutral_threshold:
            result_emotion = Emotions.NEUTRAL.value

        return result_emotion

    @staticmethod
    def is_neutral_emotion(emotion):
        """
        Checks if the EmotionResult is a neutral emotion (all emotions equal 0)

        :param emotion: EmotionResult
        :returns boolean: True if it is a neutral emotion
        """
        result = emotion[Emotions.FEAR.value] == 0 and emotion[Emotions.ANGER.value] == 0
        result = result and emotion[Emotions.JOY.value] == 0 and emotion[Emotions.ANTICIPATION.value] == 0
        result = result and emotion[Emotions.SADNESS.value] == 0 and emotion[Emotions.DISGUST.value] == 0
        result = result and emotion[Emotions.SURPRISE.value] == 0 and emotion[Emotions.TRUST.value] == 0

        return result

class Emotions(Enum):
    """
    Enumarations of Plutchnik's basic emotions
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
    NEUTRAL = 'neutral'
    