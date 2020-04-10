from enum import Enum

class EmotionResult():

    @staticmethod
    def create_emotion():
        emotion = {
           Emotions.ANGER.value: 0,
           Emotions.FEAR.value: 0,
           Emotions.SADNESS.value: 0,
           Emotions.DISGUST.value: 0,
           Emotions.JOY.value: 0,
           Emotions.ANTICIPATION.value:0,  
           Emotions.SURPRISE.value:0,
           Emotions.NEUTRAL.value:0
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
    