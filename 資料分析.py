# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:38:30 2024

@author: user
"""
import pandas as pd

df = pd.read_csv("Data.csv")
df['Date'] = pd.to_datetime(df['Date'])

total_sell = df.groupby('Item_ID')['Quantity'].sum()
sales_champion = df.groupby('Salesperson')['Quantity'].sum()
total_puchase = df.groupby('Customer_ID')['Quantity'].sum()

best_selling_item_id = total_sell.idxmax()
best_selling_quantity = total_sell.max()
print(f"Best Selling Item_ID: {best_selling_item_id}, Total all_day_Quantity = {best_selling_quantity}")

worst_selling_item_id = total_sell.idxmax()
worst_selling_quantity = total_sell.min()
print(f"Worst Selling Item_ID: {worst_selling_item_id}, Total all_day_Quantity = {worst_selling_quantity}")

best_salesperson_id = sales_champion.idxmax()
best_salesperson_quantity = sales_champion.max()
print(f"Sales Champion: {best_salesperson_id}, Total_Quantity = {best_salesperson_quantity}")

worst_salesperson_id = sales_champion.idxmin()
worst_salesperson_quantity = sales_champion.min()
print(f"Sales Champion: {worst_salesperson_id}, Total_Quantity = {worst_salesperson_quantity}")

most_purchases_customer_id = total_puchase.idxmax()
most_purchases_quantity = total_puchase.max()
print(f"Customer with Most Purchases: {most_purchases_customer_id}, Total Quantity = {most_purchases_quantity}")

least_purchases_customer_id = total_puchase.idxmin()
least_purchases_quantity = total_puchase.min()
print(f"Customer with Least Purchases: {least_purchases_customer_id}, Total Quantity = {least_purchases_quantity}")
