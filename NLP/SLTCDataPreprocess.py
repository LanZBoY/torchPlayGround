import os
import json
import pandas as pd
from NLPUtils.DataModel import Vocabulary
from NLPUtils.preprocessUtils import removePunctuation, removeWord
from nltk.corpus import stopwords
import numpy as np
from tqdm import tqdm

root_path = './NLP/SLTCDataset/'
datasets = ["20NG", "R52", "R8", 'mr', 'ohsumed_single_23']
with open('./NLP/SLTCPreprocessConfig.json') as f:
    config = json.load(f)
assert config['Dataset'] in datasets, "Dataset is not in dataset list"
DATASET = config['Dataset']
stopword_list = stopwords.words('english')
TRAIN = 'train'
TEST = 'test'
# Train Data
train_file_path = os.path.join(root_path, DATASET, f"{DATASET}_{TRAIN}.csv")
train_df = pd.read_csv(train_file_path, encoding='utf-8')
train_voc = Vocabulary(TOKENS = config["TOKEN"], MAXLEN = config['MAXLEN'])
train_data = []
for sentence in tqdm(train_df['text'], desc="Extract Training Data..."):
    sentence = removePunctuation(sentence)
    sentence = removeWord(removeWordList = stopword_list, sentence = sentence.split())
    train_voc.addWordList(sentence)
    data = [train_voc.word2idx[word] for word in sentence]
    while (len(data) < config['MAXLEN']):
        data.append(train_voc.word2idx['<UNK>'])
    train_data.append(data)
x_train = np.array(train_data, np.int64)
y_train = train_df['target'].to_numpy(dtype = np.int64)
# Test Data
test_file_path = os.path.join(root_path, DATASET, f"{DATASET}_{TEST}.csv")
test_df = pd.read_csv(test_file_path, encoding='utf-8')
test_data = []
for sentence in tqdm(test_df['text'], desc="Extract Testing Data..."):
    sentence = removePunctuation(sentence)
    sentence = removeWord(removeWordList = stopword_list, sentence = sentence.split())
    data = []
    for word in sentence:
        if train_voc.has(word):
            data.append(train_voc.word2idx[word])
        else:
            data.append(train_voc.word2idx['<UNK>'])
        if len(data) == config['MAXLEN']:
            break
    while (len(data) < config['MAXLEN']):
        data.append(train_voc.word2idx['<UNK>'])
    test_data.append(data)
x_test = np.array(test_data, np.int64)
y_test = test_df['target'].to_numpy(dtype = np.int64)

root_npy_path = os.path.join(root_path, DATASET)
x_train_path = os.path.join(root_npy_path, f"x_{TRAIN}.npy")
y_train_path = os.path.join(root_npy_path, f"y_{TRAIN}.npy")
x_test_path = os.path.join(root_npy_path, f"x_{TEST}.npy")
y_test_path = os.path.join(root_npy_path, f"y_{TEST}.npy")
np.save(x_train_path, x_train)
np.save(y_train_path, y_train)
np.save(x_test_path, x_test)
np.save(y_test_path, y_test)