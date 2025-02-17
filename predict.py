# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:13:24 2024

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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import lstm_split_data



df = pd.read_csv('min_max_normalized_data.csv', index_col = 'Date')

def split_stock_data(stock_data, label_column, delete_column, test_size = 0.3, 
                     random_state = 42):
    X = stock_data.drop([label_column, delete_column], axis = 1).values
    y = stock_data[label_column].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size, 
                                                        random_state = 
                                                        random_state)
    return X_train, X_test, y_train, y_test

label_column = 'LABEL' # 標籤欄位
delete_column = 'Next_5Day_Return' # 刪除的欄位
trainX, testX, trainY, testY = split_stock_data(df, label_column, delete_column)

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

RF = RandomForestClassifier()
RF.fit(trainX, trainY)
train_acc = RF.score(trainX, trainY)
test_acc = RF.score(testX, testY)
model_accuracies['RandomForest'] = test_acc
print(f'RF訓練集準確率 {train_acc:.2f}')
print(f'RF測試集準確率 {test_acc:.2f}')

Xgboost = XGBClassifier()
Xgboost.fit(trainX, trainY)
train_acc = Xgboost.score(trainX, trainY)
test_acc = Xgboost.score(testX, testY)
model_accuracies['XGBoost'] = test_acc
print('Xgboost訓練集準確率 %.2f' % train_acc)
print('Xgboost測試集準確率 %.2f' % test_acc)

trainX, testX, trainY, testY = \
    lstm_split_data.split_stock_data(df, label_column, delete_column)
print(f"lstm_split_data type: {type(lstm_split_data.split_stock_data(df, label_column, delete_column))}")

# 創建 LSTM 模型 
model = Sequential()
model.add(LSTM(units = 100, return_sequences = True, 
               input_shape = (trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(units = 100, return_sequences = False))
model.add(Dropout(0.3))
model.add(Dense(units = 1, activation = 'relu'))
model.compile(optimizer = Adam(learning_rate = 0.001), 
              loss = 'mean_squared_error', metrics = ['accuracy']) # 編譯模型
model.fit(trainX, trainY, epochs = 60, batch_size = 64, 
          validation_split = 0.2) # 訓練模型
# 0.001  68  64 0.554

train_loss, train_acc = model.evaluate(trainX, trainY, verbose = 0) # 訓練集準確度計算
test_loss, test_acc = model.evaluate(testX, testY, verbose = 0) # 測試集準確度計算
model_accuracies['LSTM'] = test_acc #將測試集結果儲存到字典中
print('LSTM訓練集準確率: %.4f' % train_acc)
print('LSTM測試集準確率: %.4f' % test_acc) # 0.5

#創建 GRU 模型 (GRU)
model = Sequential()
model.add(GRU(units = 100, return_sequences = True, 
              input_shape = (trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.3))
model.add(GRU(units = 100, return_sequences = False))
model.add(Dropout(0.3))
model.add(Dense(units = 1, activation = 'relu'))
model.compile(optimizer = Adam(learning_rate = 0.001), 
              loss = 'mean_squared_error', metrics = ['accuracy'])
model.fit(trainX, trainY, epochs = 10, batch_size = 32, validation_split = 0.2)
train_loss, train_acc = model.evaluate(trainX, trainY, verbose = 0)
test_loss, test_acc = model.evaluate(testX, testY, verbose = 0)
model_accuracies['GRU'] = test_acc
print('GRU訓練集準確率: %.2f' % train_acc)
print('GRU測試集準確率: %.2f' % test_acc)

# 創建 BI-LSTM 模型(Bidirectional)
model = Sequential()
model.add(Bidirectional(LSTM(units = 100, return_sequences = True), 
                        input_shape = (trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(units = 100, return_sequences = False)))
model.add(Dropout(0.3))
model.add(Dense(units = 1, activation = 'relu'))
model.compile(optimizer = Adam(learning_rate = 0.001), 
              loss ='mean_squared_error', metrics = ['accuracy'])
model.fit(trainX, trainY, epochs = 10, batch_size = 32, validation_split = 0.2)
train_loss, train_acc = model.evaluate(trainX, trainY, verbose = 0)
test_loss, test_acc = model.evaluate(testX, testY, verbose = 0)
model_accuracies['BI-LSTM'] = test_acc
print('BI-LSTM訓練集準確率: %.2f' % train_acc)
print('BI-LSTM測試集準確率: %.2f' % test_acc)

best_model = max(model_accuracies, key = model_accuracies.get)
best_accuracy = model_accuracies[best_model]
print(f'準確率最高的模型是 {best_model}，準確率為 %.2f' % best_accuracy)
