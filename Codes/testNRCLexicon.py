from nrc_emolex_utils import EmoLexHelper

lexicon = EmoLexHelper()

emotion = lexicon.get_emotion('love')

print(emotion._emotion_result)