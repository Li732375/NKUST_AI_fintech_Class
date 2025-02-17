# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:22:58 2024

@author: user
"""

import pandas as pd
stock_data = pd.read_excel('data.xlsx', index_col='Date')
columns_to_exclude = ['Next_5Day_Return', 'LABEL'] # 需要排除的欄位
columns_to_normalize = stock_data.columns.difference(columns_to_exclude) # 分離不需要正規化的欄位['Next_Day_Return', 'LABEL']

def min_max_normalize(column):
    return (column - column.min()) / (column.max() - column.min()) # 定義 Min-Max 正規化函數進行正規化

df_normalized = \
    stock_data[columns_to_normalize].apply(min_max_normalize) # 針對需要正規化的欄位進行正規化
df_final = pd.concat([df_normalized, 
                      stock_data[columns_to_exclude]], axis=1) # 合併正規化後的數據和不需要正規化的欄位
df_final.to_csv('min_max_normalized_data.csv')
print(df_final)