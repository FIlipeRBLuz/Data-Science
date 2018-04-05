# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 21:36:51 2018

@author: filipe.luz
"""

# Multiple linear regrssion

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
#Excluding the first column to don't sink on trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
#Not necessary here because the algorithm will trate it
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
#creat an object of linearregression class
regressor = LinearRegression()
#fit the multilinear regression to my train set
regressor.fit(X_train, y_train)


#Predicting the Test set results
y_pred = regressor.predict(X_test)


#Bulding the optimal model using Backward Elimination
import statsmodels.formula.api as sm
#Add a new column on 0 position with 1 value in all rows to be B0
X = np.append(arr = np.ones((50,1)).astype(int), values = X ,axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]

#ordinarie list class
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()

#verify P-value for all columns in model
regressor_OLS.summary()

#After indentify the highest p-value we
#will exclude this referencied column
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()

#3ª round , exclude the next highest p-value
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()

#4ª round , exclude the next highest p-value
X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()

#5ª round , exclude the next highest p-value
X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()
