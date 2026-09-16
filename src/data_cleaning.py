import pandas as pd
import numpy as np

df = pd.read_csv(r"D:\PROJECT\E-commerce _Return_prediction_system\Dataset\Kaggle_Ecommerce Data.csv",encoding="utf-8")
# print(df.head())
print(df.columns)
print(df.duplicated().sum())
df = df.drop_duplicates()

#convert date columns
df['order_date'] = pd.to_datetime(df['order_date'],errors="coerce")
df['delivered_date'] = pd.to_datetime(df['delivered_date'],errors="coerce")
# print(df['order_date'])

#how many days between the ordered_date to delivery_date
df['delivery_days'] = (df['delivered_date']-df['order_date']).dt.days

# cretae ordered date Fetaures
df['order_year'] = df['order_date'].dt.year
df['order_month'] = df['order_date'].dt.month
df['order_day_of_week'] = df['order_date'].dt.dayofweek   

# drop unnecessary columns 
df = df.drop(columns=["order_id","customer_id","product_id","request_date", "return_reason","order_date","delivered_date"],errors='ignore')

# convert yes ans No into binart 1:yes  0:No
df['returned'] = df['returned'].map({"No":0,"Yes":1})
#check 
print(df['returned'].value_counts())
print(df['returned'].value_counts(normalize=True)*100)
print('hello')

#check missing value
print(df.isnull().sum())

#final information
print(df.shape)
print(df.columns.tolist())

# save the cleaned datasets
df.to_csv(r"D:\PROJECT\E-commerce _Return_prediction_system\Dataset\Cleaned_Ecommerce_Data.csv",index=False) 