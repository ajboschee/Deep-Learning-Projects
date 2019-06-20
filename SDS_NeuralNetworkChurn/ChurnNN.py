# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Data
dataset = pd.read_csv('Churn_Modeling.csv')
X=dataset.iloc[:, 3:13].values
y=dataset.iloc[:, 13].values


# Encode Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Split training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 123)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

# First hidden layer and input layer
classifier.add(Dense(units=6, kernel_initializer='uniform',activation='relu', input_dim = 11))
#second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform',activation='relu'))

#output layer 
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

#compile NN with stochastic gradient descent
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#fit nn to training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

#predict test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# make confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

print(cm)


# predict new individual observation
"""
Geography: France -> 42
Credit Score: 600
Gender: Male -> 1
Age: 40
Tenure: 3
Balance: 60000
Num of Products: 2
Credit Card: Yes -> 1
Active Member: Yes -> 1
Estimated Salary: 50000
"""

# turn into 2d array and apply scale with transform
new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

print(new_prediction)


# EVALUATE, TUNE AND IMPROVE THE NUERAL NETWORK

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential() 
    classifier.add(Dense(units=6, kernel_initializer='uniform',activation='relu', input_dim = 11))
    classifier.add(Dense(units=6, kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

k_classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch=100)
accuracies = cross_val_score(estimator = k_classifier, X = X_train, y= y_train, cv=10, n_jobs=1)
mean = accuracies.mean()
variance = accuracies.std()

# Tuning the NN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential() 
    classifier.add(Dense(units=6, kernel_initializer='uniform',activation='relu', input_dim = 11))
    classifier.add(Dense(units=6, kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

gs_classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [25, 32],
              'nb_epoch': [100, 500], 
              'optimizer': ['adam','rmsprop']}

grid_search = GridSearchCV(estimator=gs_classifier, 
                           param_grid = parameters, 
                           scoring = 'accuracy', 
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_







