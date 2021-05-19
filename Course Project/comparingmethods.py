# Libraries Included:
# Numpy, Scipy, Scikit, Pandas, Sklearn

# Import python libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re, nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# nltk.download('stopwords')
# nltk.download('wordnet')

stop_nltk = set(stopwords.words('english'))

df = pd.read_json("https://raw.githubusercontent.com/mickolevm/DataFile/master/train.json")


# print(df.head(5))

# Data Cleaning
# Convert to lower case
def lower_case(tweet):
    tweet = tweet.lower()
    return tweet


# Tokenization - Converting a sentence into list of words
def tokenization(tweet):
    tweet = re.split('\W+', tweet)
    return tweet


def remove_stop_words(tweet):
    tweet = [word for word in tweet if word not in stop_nltk]
    return tweet


# Remove Punctuation
def remove_punc(tweet):
    tweet = [char for char in tweet if char not in string.punctuation]
    return tweet


sm = nltk.PorterStemmer()


def stem(tweet):
    tweet = [sm.stem(word) for word in tweet]
    return tweet


def clean_data(data):
    data = data.apply(lambda x: lower_case(x))
    data = data.apply(lambda x: tokenization(x))
    data = data.apply(lambda x: remove_stop_words(x))
    data = data.apply(lambda x: remove_punc(x))
    data = data.apply(lambda x: stem(x))

    return data


df['clean'] = clean_data(df['claim'])
df['clean_joined'] = df['clean'].apply(lambda x: ' '.join(x))

# print(df.head(10))


# Claim --> bag of words, tfidf
claims = df['clean_joined']

Tfidf = TfidfVectorizer(max_features=1500)
Count = CountVectorizer(max_features=1500)

# # print(claims)
# concatenated = []
# for i in range(len(claims)):
#     concatenated = concatenated + claims[i]
#
# print(concatenated)

Count_vectorized = Count.fit_transform(df.clean_joined)
Count_features = Count.get_feature_names()
Count_data_vect = Count_vectorized.toarray()

# df['data_count'] = list(Count_data_vect)

Tfidf_vectorized = Tfidf.fit_transform(df.clean_joined)
Tfidf_features = Tfidf.get_feature_names()
Tfidf_data_vect = pd.DataFrame(Tfidf_vectorized.toarray())

# df['data_tfidf'] = list(Tfidf_data_vect)
#
# count_features = df[['data_count', 'label']]
# tfidf_features = df[['data_tfidf', 'label']]



X = Tfidf_data_vect
y = df.label



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape)
print(y_train.shape)
# Two different training set --> with and without the claimants

# Logistic, Decision Trees, SVM, Random Forest, Naive Bayes, kNN

# num_trees = 50
# seed = 9
# scoring = "f1_macro"


# def select_model(X_train, y_train, X_test, y_test):
#     # Initializing variables for storing optimal depth and criterion
#     best_depth = 0
#     best_crit = 'gini'
#     best_accuracy = 0
#
#     # Testing accuracies of trees with 5 different depths and 2 different split criteria
#     for depth in [50]:
#         for criteria in ['gini', 'entropy']:
#             tree = RandomForestClassifier(n_estimators=50, criterion = 'entropy',max_depth=depth, random_state = 42)
#             tree.fit(X_train, y_train)
#             train_predict = tree.predict(X_train)
#             test_predict = tree.predict(X_test)
#             train_acc = f1_score(train_predict, y_train, average='macro')
#             test_acc = f1_score(test_predict, y_test,average='macro')
#             print('Using max_depth of', depth, 'and criterion of', criteria, 'the train acc was', train_acc,
#                   'and the test acc was', test_acc)
#             if train_acc > best_accuracy:
#                 best_depth = depth
#                 best_crit = criteria
#                 best_accuracy = train_acc
#                 best_tree = tree
#
#     print('The best f1score of', best_accuracy, 'was achieved with max_depth of', best_depth, 'and criterion',
#           best_crit)
#
#     return best_tree
#
#
# select_model(X_train, y_train, X_test, y_test)

# models = []
# models.append(('LR', LogisticRegression(random_state=seed)))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier(random_state=seed)))
# models.append(('RF', RandomForestClassifier(n_estimators=500, random_state=seed, max_features='sqrt')))
# models.append(('AB', AdaBoostClassifier()))
# models.append(('NB', GaussianNB()))

# train_results = []
# test_results = []
# names = []
#
# for name, model in models:
#     model.fit(X_train, y_train)
#     train_predict = model.predict(X_train)
#     test_predict = model.predict(X_test)
#     train_acc = f1_score(train_predict, y_train, average='macro')
#     test_acc = f1_score(test_predict, y_test, average='macro')
#
#     print('With model', name, ', train f1:', train_acc, ', and test f1:', test_acc)

from keras.models import Sequential
from keras import layers
from keras.preprocessing import sequence
from keras.layers import Dense, Flatten, LSTM, Conv1D, GlobalMaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.datasets import imdb
#
input_dim = X_train.shape[1]  # Number of features
#
# model = Sequential()
# model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
# model.add(layers.Dense(3, activation='softmax'))
# model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# model.summary()
# history = model.fit(X_train, y_train, epochs=10, verbose=1, validation_data=(X_test, y_test), batch_size=10)
# yhat_probs = model.predict(X_test, verbose=0)
# yhat_classes = model.predict_classes(X_test, verbose=0)
# f1 = f1_score(y_test, yhat_classes, average='macro')
# print('F1 score: %f' % f1)
# matrix = confusion_matrix(y_test, yhat_classes)
# print(matrix)
max_features = 5000
maxlen = 1500
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(3))
model.add(Activation('sigmoid'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test))
