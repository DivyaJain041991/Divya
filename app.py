# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 16:40:26 2019

@author: DJ_Home
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
Labelencoder_X_1 =LabelEncoder()
X[:,1] =Labelencoder_X_1.fit_transform(X[:,1])
Labelencoder_X_2 =LabelEncoder()
X[:,2] = Labelencoder_X_2.fit_transform(X[:,2])
onehotencoder =OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size =0.2)


import keras

from keras.models import Sequential
from keras.layers import Dense

Classifier = Sequential()

Classifier.add(Dense(units=6 , kernel_initializer='uniform', activation = 'relu', input_data =X))

Classifier.add(Dense(units=6 , kernel_initializer='uniform', activation = 'relu'))

Classifier.add(Dense(units=1 , kernel_initializer='uniform', activation = 'sigmoid'))

Classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

Classifier.fit(X_train,y_train,batch_size=10, epochs=100)

y_pred = Classifier.predict(X_test)

