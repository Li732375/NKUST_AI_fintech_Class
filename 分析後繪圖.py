# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:02:22 2024

@author: user
"""

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Data.csv")
total_sell = df.groupby("Item_ID")['Quantity'].sum()
sales_champion = df.groupby("Salesperson")['Quantity'].sum()
Customer_ID = df.groupby("Customer_ID")['Quantity'].sum().nlargest(20)

plt.figure(figsize = (12, 6))
total_sell.plot(kind = 'bar', color = 'skyblue')
plt.title('Total Sales by Item_ID')
plt.xlabel('Item_ID')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation = 45)
plt.grid(axis = 'y')
plt.show()

plt.figure(figsize = (12, 6))
sales_champion.plot(kind = 'line', color = 'green')
plt.title('Total Sales by Salesperson')
plt.xlabel('Salesperson')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation = 45)
plt.grid(axis = 'y')
plt.show()

plt.figure(figsize = (12, 6))
Customer_ID.plot(kind = 'barh', color = 'blue')
plt.title('Total Sales by Customer_ID')
plt.xlabel('Customer_ID')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation = 45)
plt.grid(axis = 'y')
plt.show()