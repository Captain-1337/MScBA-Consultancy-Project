from nrc_emolex_utils import EmoLexHelper

lexicon = EmoLexHelper()

emotion = lexicon.get_emotion('love')
# emotion joy: 1
print(emotion._emotion_result)