from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model

payments = pd.read_csv(
    r'C:\Users\Adham\Documents\Machine Learning\Machine-Learning_Assignments\ClassificationAssign\PS_20174392719_1491204439457_log.csv')

type_new = pd.get_dummies(payments['type'], drop_first=True)
payments = pd.concat([payments, type_new], axis=1)


x = payments[['step', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER', 'amount',
              'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']]
y = payments['isFraud']

scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# Create  classifier object.
lr = linear_model.LogisticRegression()

# Create StratifiedKFold object.
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
lst_accu_stratified = []

for train_index, test_index in skf.split(x, y):
    x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    lr.fit(x_train_fold, y_train_fold)
    lst_accu_stratified.append(lr.score(x_test_fold, y_test_fold))


print('List of possible accuracy:', lst_accu_stratified)
print('\nMaximum Accuracy That can be obtained from this model is:',
      max(lst_accu_stratified)*100, '%')
print('\nMinimum Accuracy:',
      min(lst_accu_stratified)*100, '%')
print('\nOverall Accuracy:',
      mean(lst_accu_stratified)*100, '%')
