import pandas as pd

def create_feature(df):
    df = df.copy()

    df["price_per_quantity"] = df["price"] / df["quantity"].replace(0,1)
    df["amount_per_quantity"] = (df["total_amount"] / df["quantity"].replace(0,1))

    df["shipping_ratio"] = (df["shipping_cost"] / df["total_amount"].replace(0,1))

    df["profit_amount"] = (df["total_amount"]*df["profit_margin"]/100)

    df["age_group"] = (pd.cut(df["customer_age"],bins=[0,25,35,50,65,100],labels=["18-25","26-35","36-50","51-65","66+"]))

    df["delivery_group"] = pd.cut(df["delivery_days"],bins=[-1,2,5,10,float("inf")],labels=["Fast","Normal","Slow","Very_Slow"])

    return df

if __name__ == "__main__":
    df = pd.read_csv(r"D:\PROJECT\E-commerce _Return_prediction_system\Dataset\Cleaned_Ecommerce_Data.csv")

    df = create_feature(df)

    print("New shape:",df.shape)
    print("\nNew columns:")
    print(df.columns.tolist())
    print("\nFeature sample:")
    print(df.head())

