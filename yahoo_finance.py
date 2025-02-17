# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:34:30 2024

@author: user
"""

import yfinance as yf
import pandas as pd

def fetch_stock_data(stockNos, start_date, end_date, interval):
    all_data = {}
    for stockNo in stockNos:
        stock_id = str(stockNo)
        data = yf.Ticker(stock_id)
        df = data.history(interval=interval, start=start_date, end=end_date)
        df.index = df.index.tz_localize(None)  # Remove timezone information
        all_data[stock_id] = df
    return all_data

stockNos = ["2330.TW", "2317.TW", "2454.TW"]  # Add more stock numbers as needed
interval = "60m"
start_date = "2024-01-15"
end_date = "2024-03-09"

stock_data = fetch_stock_data(stockNos, start_date, end_date, interval)

# Save the DataFrames to an Excel file with each stock's data in a separate sheet
output_file = "multiple_stocks_data.xlsx"
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for stock_id, df in stock_data.items():
        df.to_excel(writer, sheet_name=stock_id, index=True)

print(f"Data has been saved to {output_file}")