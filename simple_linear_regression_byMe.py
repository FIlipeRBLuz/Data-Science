#Simple linear regresion

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('Salary_Data.csv')

#creating new dataset who 
# has independent columns and values
X = dataset.iloc[:,:-1].values

#creating depend variable
y = dataset.iloc[:,1].values

#Split dataset in training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


#Feature scaling - colocando tds na mesma escala de valor
"""from sklearn.preprocessing import StandardScaler
sc_X =StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#fitting Simple liniar regresion to the training set
from sklearn.linear_model import LinearRegression
regressor =  LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test set result
y_pred = regressor.predict(X_test)


#Visualizing the traininig set results
plt.scatter(X_train, y_train, color = 'red')
#PLOT THE LINE, realize a new predict from the original data set (train) 
plt.plot(X_train,regressor.predict(X_train), color = 'blue' )
#add title 
plt.title('Salary Experience (Training Set)')
#Add label to x-axis
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#Visualizing the Test set results
#Visualizing the traininig set results
plt.scatter(X_test, y_test, color = 'red')
#PLOT THE LINE, realize a new predict from the original data set (train) 
plt.plot(X_train,regressor.predict(X_train), color = 'blue' )
#add title 
plt.title('Salary Experience (Training Set)')
#Add label to x-axis
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()




