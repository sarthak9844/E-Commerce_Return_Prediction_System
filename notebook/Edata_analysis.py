import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv(r"D:\PROJECT\E-commerce _Return_prediction_system\Dataset\Kaggle_Ecommerce Data.csv",encoding="utf-8")

print("First 5 rows:")
print(df.head())
print("\nDataset shape:")
print(df.shape)
print("\nStatistical summary:")
print(df.describe())
print("\nColumn names:")
print(df.columns)
print("\nDataset information:")
print(df.info)
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:")
print(df.duplicated().sum())
print("\nColumn names list:")
print(df.columns.tolist())
# unique value in each column
print("\nUnique values:")
for column in df.columns:
    print(df[column].nunique())

#target Distribution
print("\nReturned distribution:")
print(df['returned'].value_counts())
print(df['returned'].value_counts(normalize=True)*100)

print("\nData Types:")
print(df.dtypes)

print("\nProfit Margin Statistics:")
print(df["profit_margin"].describe())

print("\nHighest Profit Margin:")
print(df["profit_margin"].nlargest(10))

print("\nLowest Profit Margin:")
print(df["profit_margin"].nsmallest(10))

print("\nOrders with highest profit margin:")
print(df.nlargest(10, "profit_margin")[["price","quantity","discount","total_amount","shipping_cost","profit_margin","returned"]])
