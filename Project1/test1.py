import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report , r2_score

data = pd.read_csv('C:/Users/coolk/OneDrive/Desktop/stock/GOOGL_2006-01-01_to_2018-01-01.csv')
data = data.drop('Name',axis=1)
data1 = data
X_1 = data.drop('Close',axis=1)
Y_1 = data['Close']

date_int = []
for i in range(len(data)):
    date_int.append(data['Date'][i].rstrip().split('-'))
    data['Date'][i] = ''.join(date_int[i])

X = data.drop('Close',axis=1)
Y = data['Close']

def train_randomforest(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.33, random_state=42)
    rfg = RandomForestRegressor()
    rfg.fit(X_train, Y_train)
    Y_pred = rfg.predict(X_test)
    #print(Y_pred)
    Y_diff = Y_pred - Y_test
    #print(Y_diff)
    print("Accuracy of Random Forest Model :",r2_score(Y_test, Y_pred))
    plt.figure(figsize=(16, 8))
    plt.plot(Y_pred, '-b', label='Close Price History')

    plt.figure(figsize=(16, 8))
    plt.plot(Y_diff, '-g', label='Close Price History')

    plt.figure(figsize=(16, 8))
    plt.plot(Y_test, '-r', label='Close Price History')

    plt.show();

train_randomforest(X,Y)


def _train_KNN(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.33, random_state=42)
    knn = KNeighborsRegressor()
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    #print(Y_pred)
    Y_diff = Y_pred - Y_test
    # print(Y_diff)
    # print(confusion_matrix(Y_test.astype(int), Y_pred.astype(int)))
    print("Accuracy of KNN Model :",r2_score(Y_test, Y_pred))
    plt.figure(figsize=(16, 8))
    plt.plot(Y_pred, '-b', label='Close Price History')

    plt.figure(figsize=(16, 8))
    plt.plot(Y_diff, '-g', label='Close Price History')

    plt.figure(figsize=(16, 8))
    plt.plot(Y_test, '-r', label='Close Price History')

    plt.show()


_train_KNN(X,Y)