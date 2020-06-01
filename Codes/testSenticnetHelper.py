from senticnet_utils import SenticNetHelper

pos1 = {}
pos1["word"] = "love"
pos1["pos"] = "V"
pos1["order"] = 1
pos2 = {}
pos2["word"] = "you"
pos2["pos"] = "PRP"
pos2["order"] = 2
snh = SenticNetHelper()

pos = [pos1,pos2]
emotion = snh.get_emotion('love you',pos)
# love_person?
print(emotion.to_string())


pos1 = {}
pos1["word"] = "love"
pos1["pos"] = "V"
pos1["order"] = 1
pos2 = {}
pos2["word"] = "Roger"
pos2["pos"] = "PRP"
pos2["order"] = 2
pos2["ne"] = "PERSON"
snh = SenticNetHelper()

pos = [pos1,pos2]
emotion = snh.get_emotion('love Roger',pos)
# love_person?
print(emotion.to_string())