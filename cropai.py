import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Dense, Dropout, Activation, Flatten


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingRegressor

from flask import Flask, render_template

path = "C:\\Users\\savit\\Downloads\\yield_df.csv\\yield_df.csv"
data = pd.read_csv(path)
data_categorical = data[['Area']]
dummies = pd.get_dummies(data_categorical)
data = data.drop(['Area'], axis = 1)
data = pd.concat([data, dummies], axis=1)
data_categorical = data[['Item']]
dummies = pd.get_dummies(data_categorical)
data = data.drop(['Item'], axis = 1)
data = pd.concat([data, dummies], axis=1)
data = data.drop(['Unnamed: 0'],axis = 1)
data.shape
Y = data['hg/ha_yield']
X = data.drop(['hg/ha_yield'],axis = 1)
X = X.drop(['Year'],axis = 1)
# do one hot encoding

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(X_test.shape)
model = BaggingRegressor(n_estimators=150, random_state=42)
name = 'Bagging Regressor'
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
MSE = mean_squared_error(y_test, y_pred)
R2_score = r2_score(y_test, y_pred)
acc = (model.score(X_train , y_train)*100)
print(f'The accuracy of the {name} Model Train is {acc:.2f}')
acc =(model.score(X_test , y_test)*100)
print(f'The accuracy of the  {name} Model Test is {acc:.2f}')
