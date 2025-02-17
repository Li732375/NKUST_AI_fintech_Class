# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:06:27 2024

@author: user
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

df = pd.read_excel('I005.xlsx', index_col = 'Year_Week')

def split_stock_data(stock_data, label_column, delete_column, test_size = 0.3, 
                     random_state = 42):
    X = stock_data.drop([label_column] + delete_column, axis = 1).values
    y = stock_data[label_column].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=
                                                        random_state)
    return X_train, X_test, y_train, y_test

label_column = 'LABEL'
delete_column = ['sell_Return', 'Item_ID', 'Quantity', 'Unit_price']


trainX, testX, trainY, testY = split_stock_data(df, label_column, 
                                                delete_column)

#print(type(split_stock_data(df, label_column, delete_column)))
#print(len(split_stock_data(df, label_column, delete_column)))
ones_count = df['LABEL'].sum() #計算標註為1(上漲)的資料
zeros_count = len(df) - ones_count  #計算標註為0(下跌)的資料
print(f"1(上漲): {ones_count}")
print(f"0(下跌): {zeros_count}")

model_accuracies = {}

KNN = KNeighborsClassifier()
KNN.fit(trainX, trainY)
train_acc = KNN.score(trainX, trainY)
test_acc = KNN.score(testX, testY)
model_accuracies['KNN'] = test_acc
print(f'KNN訓練集準確率 {train_acc:.2f}')
print('KNN測試集準確率 %.2f' % test_acc)

SVM = svm.SVC()
SVM.fit(trainX, trainY)
train_acc = SVM.score(trainX, trainY)
test_acc = SVM.score(testX, testY)
model_accuracies['SVM'] = test_acc
print(f'SVM訓練集準確率 {train_acc:.2f}')
print(f'SVM驗證集準確率 {test_acc:.2f}')

Logistic = LogisticRegression()
Logistic.fit(trainX, trainY)
train_acc = Logistic.score(trainX, trainY)
test_acc = Logistic.score(testX, testY)
model_accuracies['LogisticRegression'] = test_acc
print(f'LR訓練集準確率 {train_acc:.2f}')
print(f'LR測試集準確率 {test_acc:.2f}')

Bayesian = GaussianNB()
Bayesian.fit(trainX, trainY)
train_acc = Bayesian.score(trainX, trainY)
test_acc = Bayesian.score(testX, testY)
model_accuracies['GaussianNB'] = test_acc
print(f'Bayes訓練集準確率 {train_acc:.2f}')
print(f'Bayes測試集準確率 {test_acc:.2f}')
excel = pd.DataFrame(data = model_accuracies, 
                     index = [0]).T.to_excel('model_accuracies.xlsx')

RF = RandomForestClassifier()
RF.fit(trainX, trainY)
train_acc = RF.score(trainX, trainY)
test_acc = RF.score(testX, testY)
model_accuracies['RandomForest'] = test_acc
print(f'RF訓練集準確率 {train_acc:.2f}')
print(f'RF測試集準確率 {test_acc:.2f}')

# important feature Analysis
import ImportFeatureAnalysis
ImportFeatureAnalysis.visualize_feature_importance('I005_含中文字.xlsx')

Xgboost = XGBClassifier()
Xgboost.fit(trainX, trainY)
train_acc = Xgboost.score(trainX, trainY)
test_acc = Xgboost.score(testX, testY)
model_accuracies['XGBoost'] = test_acc
print('Xgboost訓練集準確率 %.2f' % train_acc)
print('Xgboost測試集準確率 %.2f' % test_acc)

best_model = max(model_accuracies, key=model_accuracies.get) # 找出準確率最高的模型
best_accuracy = model_accuracies[best_model] # 找出準確率最高的模型的名稱
print(f'準確率最高的模型是 {best_model}，準確率為 %.2f' % best_accuracy)
