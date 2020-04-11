from enum import Enum

class EmotionResult():

    @staticmethod
    def create_emotion(anger = 0, fear = 0, sadness = 0, disgust = 0, joy = 0, anticipation = 0, surprise = 0, neutral = 0):
        """
        Creates an EmotionResult vector
        """
        emotion = {
           Emotions.ANGER.value: anger,
           Emotions.FEAR.value: fear,
           Emotions.SADNESS.value: sadness,
           Emotions.DISGUST.value: disgust,
           Emotions.JOY.value: joy,
           Emotions.ANTICIPATION.value:anticipation,  
           Emotions.SURPRISE.value:surprise,
           Emotions.NEUTRAL.value:neutral
        }
        return emotion

    @staticmethod
    def create_neutral_emotion():
        emotion = EmotionResult.create_emotion()
        emotion[Emotions.NEUTRAL.value] = 1
        return emotion

class Emotions(Enum):
    """
        Enumarations of Plutchnik's basic emotions plus neutral
    """
    ANGER = 'anger'
    FEAR = 'fear'
    SADNESS = 'sadness'
    DISGUST = 'disgust'
    JOY = 'joy'
    TRUST = 'trust'
    ANTICIPATION = 'anticipation'
    SURPRISE = 'surprise'
    NEUTRAL = 'neutral'
    