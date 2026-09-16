import pandas as pd
from feature_engineering import create_feature

df = pd.read_csv(r"D:\PROJECT\E-commerce _Return_prediction_system\Dataset\Cleaned_Ecommerce_Data.csv",encoding="utf-8")

df = create_feature(df)

x = df.drop("returned",axis=1)
y = df['returned']
print(x.dtypes)
print(y.dtypes)

# features type
categorical_features = ["category","payment_method","region","customer_gender","age_group","delivery_group"]

numerical_features = ["price","discount","quantity","total_amount","shipping_cost","profit_margin","customer_age","delivery_days","order_year","order_month","order_day_of_week","price_per_quantity","amount_per_quantity","shipping_ratio","profit_amount"]



# train test split the data
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42,stratify=y)

#create preprocessor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing  import OneHotEncoder,StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
       ("numerical",StandardScaler(),numerical_features),
       ("categorical",OneHotEncoder(handle_unknown="ignore"),categorical_features)
    ]
)

# display information

print("PREPROCESSING INFORMATION\n")
print("Total rows:",len(df))
print("\nfeatures",x.shape[1])
print("\nTraining sample:",len(x_train))
print("\nTesting sample:",len(x_test))
print("\nnumerical features\n",numerical_features)
print("\ncategorical features\n",categorical_features)
print("\nTraining target:",y_train.value_counts())
print("\nTesting target:",y_test.value_counts())

#test processor
x_train_processed = preprocessor.fit_transform(x_train)
x_test_processed = preprocessor.transform(x_test)

print("\noriginal training shape:",x_train.shape)
print("\nprocessed training shape:",x_train_processed.shape)
print("\nprocessed testing shape:",x_test.shape)
print("\nprocessed testing shape:",x_test_processed.shape)