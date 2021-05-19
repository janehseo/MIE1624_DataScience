# Libraries Included:
# Numpy, Scipy, Scikit, Pandas

import numpy as np
import pandas as pd
import re, nltk
import string
import random
import json
import os
from random import randint
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from nltk.corpus import stopwords
from sklearn.naive_bayes import BernoulliNB

#nltk.download('stopwords')
#nltk.download('wordnet')

# TRAINING_FILEPATH = '/usr/local/dataset/training.json'
# TRAINING_FILEPATH = "https://raw.githubusercontent.com/mickolevm/DataFile/master/train.json"
# METADATA_FILEPATH = '/usr/local/dataset/metadata.json'
METADATA_FILEPATH = "https://raw.githubusercontent.com/mickolevm/DataFile/master/train.json"
# ARTICLES_FILEPATH = '/usr/local/dataset/articles'
# PREDICTION_FILEPATH = '/usr/local/predictions.txt'
PREDICTION_FILEPATH = 'predictions.txt'
CLAIMANT_FILEPATH = 'claimants.txt'
VECTORIZER_FILEPATH = 'vectorizer.pickle'
MODEL_FILEPATH = 'model.pickle'



# df = pd.read_json(TRAINING_FILEPATH)
test = pd.read_json(METADATA_FILEPATH)

# Stop Words in nltk
stop_nltk = set(stopwords.words('english'))


# Convert to lower case
def lower_case(claim):
    claim = claim.lower()
    return claim


# Tokenization - Converting a sentence into list of words
def tokenization(claim):
    claim = re.split('\W+', claim)
    return claim


def remove_stop_words(claim):
    claim = [word for word in claim if word not in stop_nltk]
    return claim


# Remove Punctuation
def remove_punc(claim):
    claim = [char for char in claim if char not in string.punctuation]
    return claim


sm = nltk.PorterStemmer()


def stem(claim):
    claim = [sm.stem(word) for word in claim]
    return claim


def clean_data(data):
    data = data.apply(lambda x: lower_case(x))
    data = data.apply(lambda x: tokenization(x))
    data = data.apply(lambda x: remove_stop_words(x))
    data = data.apply(lambda x: remove_punc(x))
    data = data.apply(lambda x: stem(x))

    return data


# Preprocess Data
# df['clean'] = clean_data(df['claim'])
# df['clean_joined'] = df['clean'].apply(lambda x: ' '.join(x))

test['clean'] = clean_data(test['claim'])
test['clean_joined'] = test['clean'].apply(lambda x: ' '.join(x))

# Vectorize claims

# Fit and Transform Train data
# tf_vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None, analyzer='word', max_features=1200)

# train_data_features = tf_vectorizer.fit_transform(df['clean_joined'])
# train_data_features = train_data_features.toarray()
# pickle.dump(tf_vectorizer, open("vectorizer.pickle", "wb"))
#tf_vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
tf_vectorizer = pickle.load(open(VECTORIZER_FILEPATH, "rb"))

feature_names = tf_vectorizer.get_feature_names()

# claim = pd.DataFrame(train_data_features, columns=feature_names, index=df.index)

# Transform test data
test_data_features = tf_vectorizer.transform(test['clean_joined'])
test_data_features = test_data_features.toarray()

claim_test = pd.DataFrame(test_data_features, columns=feature_names, index=test.index)

# Train claimant cleaning
# claimant_dummies = pd.get_dummies(df['claimant'])
# claimant = claimant_dummies.loc[:, (claimant_dummies == 0).mean() < .999]
# pickle.dump(claimant.columns.tolist(), open("claimants.txt", "wb"))
# claimant = pickle.load(open("claimants.txt", "rb"))
claimant = pickle.load(open(CLAIMANT_FILEPATH, "rb"))

# Test claimant cleaning
claimant_test_dummies = pd.get_dummies(test['claimant'])
claimant_test = pd.DataFrame()

for column in claimant:
    if column in claimant_test_dummies.columns:
        claimant_test[column] = claimant_test_dummies[column]
        print('True')
    else:
        claimant_test[column] = 0
        print('False')

# Train Data
# X = pd.concat([claim, claimant], axis=1)
# y = df['label'].copy()

# Test Data

test_data = pd.concat([claim_test, claimant_test], axis=1)

# BNB
# model = BernoulliNB().fit(X, y)

# pickle.dump(model, open("model.pickle", "wb"))
model = pickle.load(open(MODEL_FILEPATH, "rb"))


y_predict = model.predict(test_data)
predict_ids = test_data.index.values

f = open(PREDICTION_FILEPATH, "w+")


for ii in range(y_predict.size):
    f.write('%d,%d\n' % (test['id'][ii], y_predict[ii]))
f.close()

print('Finished writing predictions.')
