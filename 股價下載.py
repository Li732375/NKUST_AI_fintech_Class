# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:33:10 2024

@author: user
"""

import yfinance as yf
import matplotlib.pyplot as plt
stock_symbol = '2330.TW' # 輸入股票代號下載股價資料
stock_data = yf.download(stock_symbol, start='2010-01-01', end='2024-05-01') # 獲取特定日期範圍的股票資料
excel_filename = f'{stock_symbol}_stock_data.xlsx' # 將股票資料存儲為 Excel 檔案，以股票代號作為檔案名稱
stock_data.to_excel(excel_filename)
print(f"股票資料已存儲為 '{excel_filename}'")
print(stock_data)
stock_data['Close'].plot() # 用stock_data的CLOSE畫出圖形
plt.xlabel("Date") #x軸的標籤
plt.ylabel("Closing Price") #y軸的標籤
plt.title(f"{stock_symbol} Stock Price") #圖標題
plt.show()