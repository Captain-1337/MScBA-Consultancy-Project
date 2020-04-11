from enum import Enum
from random import random

class EmotionResult():

    @staticmethod
    def create_emotion(anger = 0, fear = 0, sadness = 0, disgust = 0, joy = 0, trust = 0, anticipation = 0, surprise = 0, neutral = 0):
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
           Emotions.NEUTRAL.value:neutral
        }
        return emotion

    @staticmethod
    def create_neutral_emotion():
        emotion = EmotionResult.create_emotion(neutral=1)
        return emotion

    @staticmethod
    def create_random_emotion():
        if random() < 0.9:
            emotion = EmotionResult.create_emotion(anger = random(), fear = random(), sadness = random(), disgust = random(), joy = random(), trust = random(), anticipation = random(), surprise = random(), neutral = 0)
        else:
            emotion = EmotionResult.create_neutral_emotion()
        
        return emotion

    @staticmethod
    def get_primary_emotion(emotion):
        """
            Get the primary emotion out of an EmotionResult
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
        if float(emotion[Emotions.NEUTRAL.value]) > emotion_intensity:
            result_emotion = Emotions.NEUTRAL.value
            emotion_intensity = float(emotion[Emotions.NEUTRAL.value])
        
        return result_emotion
    

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
    