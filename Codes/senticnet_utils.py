from enum import Enum
from senticnet.senticnet import SenticNet
from emotion_utils import Emotions, EmotionResult

class SenticNetHelper():
    _sn = SenticNet()

    def __init__(self):
        None

    def get_emotion(self, concept):
        """
        Gets from a word or word concept an EmtotionResult vector
        """
        concept = concept.replace(" ", "_")
        if concept in self._sn.data:
            sentics = self._sn.sentics(concept)        
            emotion = self._get_emotion_from_sentics(sentics)
        else:
            emotion = EmotionResult.create_neutral_emotion()
        return emotion

    def _get_emotion_from_sentics(self, sentics):
        """
        Gets an EmotionResult and translates a sentics vector into an emotion vector

        'pleasantness_value', 
            => +:joy  -:sadness
        'attention_value'
            => +:anticipation  -:surprise
        'sensitivity_value', 
            => +:anger  -:fear
        'aptitude_value'
            => +:trust  -:disgust
        # interest => anticipation admiration => trust
        """
        emotion = EmotionResult.create_emotion()

        if str(sentics.get("pleasantness")).isnumeric:
            pleasantness_value = float(sentics.get("pleasantness"))
        else:
            pleasantness_value = 0

        if str(sentics.get("attention")).isnumeric:
            attention_value = float(sentics.get("attention"))
        else:
            attention_value = 0

        if str(sentics.get("sensitivity")).isnumeric:
            sensitivity_value = float(sentics.get("sensitivity"))
        else:
            sensitivity_value = 0

        if str(sentics.get("aptitude")).isnumeric:
            aptitude_value = float(sentics.get("aptitude"))
        else:
            aptitude_value = 0

        # pleasantness for joy or sadness
        if pleasantness_value > 0:
            emotion["joy"] = pleasantness_value
            emotion["sadness"] = 0
        elif pleasantness_value < 0:
            emotion["sadness"] = abs(pleasantness_value)
            emotion["joy"] = 0
        else:
             emotion["sadness"] = 0
             emotion["joy"] = 0

        # attention for anticipation or surprise
        if attention_value > 0:
            emotion["anticipation"] = attention_value
            emotion["surprise"] = 0
        elif attention_value < 0:
            emotion["surprise"] = abs(attention_value)
            emotion["anticipation"] = 0
        else:
             emotion["surprise"] = 0
             emotion["anticipation"] = 0

        # sensitivity for anger or fear
        if sensitivity_value > 0:
            emotion["anger"] = sensitivity_value
            emotion["fear"] = 0
        elif sensitivity_value < 0:
            emotion["fear"] = abs(sensitivity_value)
            emotion["anger"] = 0
        else:
             emotion["fear"] = 0
             emotion["anger"] = 0

        # aptitude for trust or disgust
        if aptitude_value > 0:
            emotion["trust"] = aptitude_value
            emotion["disgust"] = 0
        elif aptitude_value < 0:
            emotion["disgust"] = abs(aptitude_value)
            emotion["trust"] = 0
        else:
             emotion["disgust"] = 0
             emotion["trust"] = 0

        return emotion

    @staticmethod
    def reduce_emotion_to_sentics_dimensions(emotion):
        """
        Reduces the emotion so it can have only one of the two opposite emotions
            'pleasantness_value', 
                => +:joy  -:sadness
            'attention_value'
                => +:anticipation  -:surprise
            'sensitivity_value', 
                => +:anger  -:fear
            'aptitude_value'
            => +:trust  -:disgust
        """
        reduced_emotion = emotion

        # pleasantness_value
        value = emotion[Emotions.JOY.value] - emotion[Emotions.SADNESS.value]
        if value > 0:
            reduced_emotion[Emotions.JOY.value] = value
            reduced_emotion[Emotions.SADNESS.value] = 0
        else:
            reduced_emotion[Emotions.JOY.value] = 0
            reduced_emotion[Emotions.SADNESS.value] = value

        # attention_value
        value = emotion[Emotions.ANTICIPATION.value] - emotion[Emotions.SURPRISE.value]
        if value > 0:
            reduced_emotion[Emotions.ANTICIPATION.value] = value
            reduced_emotion[Emotions.SURPRISE.value] = 0
        else:
            reduced_emotion[Emotions.ANTICIPATION.value] = 0
            reduced_emotion[Emotions.SURPRISE.value] = value

        # sensitivity_value
        value = emotion[Emotions.ANGER.value] - emotion[Emotions.FEAR.value]
        if value > 0:
            reduced_emotion[Emotions.ANGER.value] = value
            reduced_emotion[Emotions.FEAR.value] = 0
        else:
            reduced_emotion[Emotions.ANGER.value] = 0
            reduced_emotion[Emotions.FEAR.value] = value

        # aptitude_value
        value = emotion[Emotions.TRUST.value] - emotion[Emotions.DISGUST.value]
        if value > 0:
            reduced_emotion[Emotions.TRUST.value] = value
            reduced_emotion[Emotions.DISGUST.value] = 0
        else:
            reduced_emotion[Emotions.TRUST.value] = 0
            reduced_emotion[Emotions.DISGUST.value] = value      

        return reduced_emotion
        
    @staticmethod
    def negatate_emotion(emotion):
        """
        Negatate an emotion by a fixed value of 0.8 to the opposite emotion (1 is max value)
        e.g. reduce anger:0.6 by 0.8 will become fear:0.2
        or reduce anger:1 by 0.8 will become anger:0.2
        Only negatate emotions with certain intesity to avoid strange results
            => emotions above 0.4
        """
        min_value = 0.4
        reduce_value = 0.8
        temp_value = 0
        
        # negotation the emotion only can have one of the 4 emotion pleasentnes
        neg_emotion = SenticNetHelper.reduce_emotion_to_sentics_dimensions(emotion)
        # ANGER - FEAR
        if neg_emotion[Emotions.ANGER.value] > min_value:
            # ANGER
            temp_value = neg_emotion[Emotions.ANGER.value] - reduce_value
            if temp_value < 0:
                neg_emotion[Emotions.FEAR.value] = abs(temp_value)
                neg_emotion[Emotions.ANGER.value] = 0
            else:
                neg_emotion[Emotions.ANGER.value] = abs(temp_value)
        elif neg_emotion[Emotions.FEAR.value] > min_value:
            # FEAR
            temp_value = neg_emotion[Emotions.FEAR.value] - reduce_value
            if temp_value < 0:
                neg_emotion[Emotions.ANGER.value] = abs(temp_value)
                neg_emotion[Emotions.FEAR.value] = 0
            else:
                neg_emotion[Emotions.FEAR.value] = abs(temp_value)
        else :
            # set to zero if no emotion is above min_value 
            neg_emotion[Emotions.ANGER.value] = 0
            neg_emotion[Emotions.FEAR.value] = 0

        # JOY - SADNESS
        if neg_emotion[Emotions.JOY.value] > min_value:
            # JOY
            temp_value = neg_emotion[Emotions.JOY.value] - reduce_value
            if temp_value < 0:
                neg_emotion[Emotions.SADNESS.value] = abs(temp_value)
                neg_emotion[Emotions.JOY.value] = 0
            else:
                neg_emotion[Emotions.JOY.value] = abs(temp_value)
        elif neg_emotion[Emotions.SADNESS.value] > min_value:
            # SADNESS
            temp_value = neg_emotion[Emotions.SADNESS.value] - reduce_value
            if temp_value < 0:
                neg_emotion[Emotions.JOY.value] = abs(temp_value)
                neg_emotion[Emotions.SADNESS.value] = 0
            else:
                neg_emotion[Emotions.SADNESS.value] = abs(temp_value)
        else :
            # set to zero if no emotion is above min_value 
            neg_emotion[Emotions.JOY.value] = 0
            neg_emotion[Emotions.SADNESS.value] = 0

        # ANICIPATION - SURPRISE
        if neg_emotion[Emotions.ANTICIPATION.value] > min_value:
            # ANICIPATION
            temp_value = neg_emotion[Emotions.ANTICIPATION.value] - reduce_value
            if temp_value < 0:
                neg_emotion[Emotions.SURPRISE.value] = abs(temp_value)
                neg_emotion[Emotions.ANTICIPATION.value] = 0
            else:
                neg_emotion[Emotions.ANTICIPATION.value] = abs(temp_value)
        elif neg_emotion[Emotions.SURPRISE.value] > min_value:
            # SURPRISE
            temp_value = neg_emotion[Emotions.SURPRISE.value] - reduce_value
            if temp_value < 0:
                neg_emotion[Emotions.ANTICIPATION.value] = abs(temp_value)
                neg_emotion[Emotions.SURPRISE.value] = 0
            else:
                neg_emotion[Emotions.SURPRISE.value] = abs(temp_value)
        else :
            # set to zero if no emotion is above min_value 
            neg_emotion[Emotions.ANTICIPATION.value] = 0
            neg_emotion[Emotions.SURPRISE.value] = 0

        # TRUST - DISGUST
        if neg_emotion[Emotions.TRUST.value] > min_value:
            # TRUST
            temp_value = neg_emotion[Emotions.TRUST.value] - reduce_value
            if temp_value < 0:
                neg_emotion[Emotions.DISGUST.value] = abs(temp_value)
                neg_emotion[Emotions.TRUST.value] = 0
            else:
                neg_emotion[Emotions.TRUST.value] = abs(temp_value)
        elif neg_emotion[Emotions.DISGUST.value] > min_value:
            # DISGUST
            temp_value = neg_emotion[Emotions.DISGUST.value] - reduce_value
            if temp_value < 0:
                neg_emotion[Emotions.TRUST.value] = abs(temp_value)
                neg_emotion[Emotions.DISGUST.value] = 0
            else:
                neg_emotion[Emotions.DISGUST.value] = abs(temp_value)
        else :
            # set to zero if no emotion is above min_value 
            neg_emotion[Emotions.TRUST.value] = 0
            neg_emotion[Emotions.DISGUST.value] = 0

        return neg_emotion


        





