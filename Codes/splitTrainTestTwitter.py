from corpora_utils import CorporaHelper,CorporaDomains, CorporaProperties
import numpy as np
import os
"""
Split twitter data in Train and Test set
"""

corpora_helper = CorporaHelper("emoint_all_emotions.csv", separator='\t')
corpora_helper.remove_cleaned_corpus()
corpora_helper.remove_column('intensity')
corpora_helper.add_domain('twitter')

count_joy = 0
count_sadness = 0
count_anger = 0
count_fear = 0
max_per_emotion = 2000

corpora_helper.remove_emotion("other")
corpora_helper.remove_emotion("noemotion")
corpora_helper.remove_emotion("trust")
corpora_helper.remove_emotion("anticipation")
corpora_helper.remove_emotion("surprise")
corpora_helper.remove_emotion("disgust")

# reduce the data to have n per emotions
corpora_helper.equalize_data_emotions(max_per_emotion)

# one more shuffle
corpora_helper.shuffle_data()
print('number of anger labels: ', corpora_helper.get_emotion_data('anger').shape[0])
print('number of fear labels: ', corpora_helper.get_emotion_data('fear').shape[0])
print('number of joy labels: ', corpora_helper.get_emotion_data('joy').shape[0])
print('number of sadness labels: ', corpora_helper.get_emotion_data('sadness').shape[0])
max_data = count_anger + count_fear + count_joy + count_sadness
print('--------------------------------------------------------------------')

# Split 90% for train an avlidation and 10% for test
train, test = corpora_helper.split_train_test_data(0.9)

# Train/validate Dataset
train_corpora_helper = CorporaHelper()
train_corpora_helper.set_data(train)
trainfilepath = 'corpora/twitter_2000_train.csv'
train_corpora_helper.write_to_csv(trainfilepath,sep='\t')
print('number of anger labels train: ', train_corpora_helper.get_emotion_data('anger').shape[0])
print('number of fear labels train: ', train_corpora_helper.get_emotion_data('fear').shape[0])
print('number of joy labels train: ', train_corpora_helper.get_emotion_data('joy').shape[0])
print('number of sadness labels train: ', train_corpora_helper.get_emotion_data('sadness').shape[0])
print('--------------------------------------------------------------------')
print(f'Train dataset has been stored to: {trainfilepath}')

# Test Data set
test_corpora_helper = CorporaHelper()
test_corpora_helper.set_data(test)
testfilepath = 'corpora/twitter_2000_test.csv'
test_corpora_helper.write_to_csv(testfilepath,sep='\t')
print('number of anger labels test: ', test_corpora_helper.get_emotion_data('anger').shape[0])
print('number of fear labels test: ', test_corpora_helper.get_emotion_data('fear').shape[0])
print('number of joy labels test: ', test_corpora_helper.get_emotion_data('joy').shape[0])
print('number of sadness labels test: ', test_corpora_helper.get_emotion_data('sadness').shape[0])
print('--------------------------------------------------------------------')
print(f'Test dataset has been stored to: {testfilepath}')