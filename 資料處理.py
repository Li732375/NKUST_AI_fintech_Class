# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:06:56 2024

@author: user
"""

import pandas as pd


df = pd.read_csv("Data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by = ['Date'], inplace = True)
df.set_index('Date', inplace = True)
df.drop(['Customer_ID', 'Customer_name', 'Industry', 'Item_name', 
         'Salesperson'], axis = 1, inplace = True)

df['Year_Week'] = df.index.to_series().apply(lambda x: 
                                             f"{x.isocalendar()[0]} - \
                                             W{x.isocalendar()[1]:02}")
df = df.sort_values(by = ["Item_ID", "Year_Week"])

df['weely_Quantity'] = df.groupby(['Item_ID', 
                                   'Year_Week'])['Quantity'].transform('sum')
df['weekly_mean_Unit_price'] = df.groupby(['Item_ID', 
                                   'Year_Week'])['Unit_price'].transform('mean')
df['weely_count'] = df.groupby(['Item_ID', 
                                   'Year_Week'])['Quantity'].transform('count')

df = df.drop_duplicates(subset=["Item_ID", "Year_Week"])

for i in range(1, 6):
    df[f'weely_Quantity_{i}'] = \
        df.groupby(["Item_ID"])['weely_Quantity'].shift(i)

df['sell_Return'] = df.groupby(["Item_ID"])['weely_Quantity'].diff(1).shift(-1)

def classify_return(x):
    return 1 if x > 0 else 0

df['LABEL'] = df['sell_Return'].apply(classify_return)

df = df.dropna()

for item_id, group in df.groupby('Item_ID'):
    filename = f"{item_id}.xlsx"
    group.to_excel(filename, index = False)
    print(f"文件 {filename} 已成功保存，資料形狀為 {group.shape}。")
print("文件已成功保存")
