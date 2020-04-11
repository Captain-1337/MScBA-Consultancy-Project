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


