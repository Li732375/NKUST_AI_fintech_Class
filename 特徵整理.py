# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:36:11 2024

@author: user
"""
import pandas as pd
import talib

stock_data = pd.read_excel('2330.TW_stock_data.xlsx', index_col='Date')  # 讀取股價資料

missing_values = stock_data.isnull().sum() #檢查每一列是否有空值

print(missing_values)

stock_data.drop(columns=['Adj Close'], inplace=True)
df_close = stock_data['Close']
#處理x資料
stock_data['MA_20'] = talib.SMA(df_close,20) # 計算MA20
stock_data['RSI_14'] = talib.RSI(df_close,14)  # 計算RSI
macd, macdsignal, macdhist = talib.MACD(df_close, fastperiod=12, slowperiod=26, signalperiod=9)  # 計算MACD
stock_data['MACD'] = macd #將MACD計算結果存回資料中

columns_to_shift = ['Close','MA_20','RSI_14','MACD'] #選取需要進行處理的欄位名稱

for period in range(5, 21,5): #運用迴圈帶入前N期收盤價
        for column in columns_to_shift: #運用迴圈走訪所選的欄位名稱
            stock_data[f'{column}_{period}'] = stock_data[column].shift(period) #運用.shift()方法取得收盤價

#處理y資料
stock_data['Next_5Day_Return'] = stock_data['Close'].diff(5).shift(-5)   #計算價格變化

def classify_return(x):
    return 1 if x > 0 else 0  # 標示漲跌，大於0標示為漲(1)，小於0標示為跌(0)

stock_data['LABEL'] = stock_data['Next_5Day_Return'].apply(classify_return)  # 創造新的一列LABEL來記錄漲跌
stock_data = stock_data.dropna()  # 刪除因技術指標計算出現的空值
stock_data.to_excel("data.xlsx")  #將整理好的資料存成excel

ones_count = (stock_data['LABEL'] == 1).sum()
zero_count = (stock_data['LABEL'] == 0).sum()
print(f"上漲數為 {ones_count}")
print(f"下跌數為 {zero_count}")
