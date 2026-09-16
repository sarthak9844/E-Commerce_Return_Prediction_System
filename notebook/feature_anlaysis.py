import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\PROJECT\E-commerce _Return_prediction_system\Dataset\Cleaned_Ecommerce_Data.csv",encoding="utf-8")

print("\nDataset shape:")
print(df.shape)
            # return rate by category
category_return = (
    df.groupby("category")["returned"].agg(["count", "sum", "mean"]).sort_values("mean", ascending=False)
)
category_return["return_rate"] = (category_return["mean"] * 100)
print(category_return)


                # return rate by payment method
payment_return = (
    df.groupby("payment_method")["returned"].agg(["count","sum","mean"])
)
payment_return["return_rate"] = (payment_return["mean"]*100)

                # return rate by region
region_return = (
    df.groupby("region")["returned"].agg(["count","sum","mean"])
)
region_return["return_rate"] = (region_return["mean"]*100)

                # return rate by gender
gender_return = (
    df.groupby("customer_gender")["returned"].agg(["count","sum","mean"])
)
gender_return["return_rate"] = (gender_return["mean"]*100)

                # return rate by discount
discount_return=(
    df.groupby("discount")["returned"].agg(["count","sum","mean"])
)
discount_return["return_rate"]=(discount_return["mean"]*100)

                # return rate by month
month_return=(
    df.groupby("order_month")["returned"].agg(["count","sum","mean"])
)
month_return["return_rate"] = (month_return["mean"]*100)

                # Return rate by delivery days
delivery_return = (
    df.groupby("delivery_days")["returned"].agg(["count", "sum","mean"]).sort_values("mean", ascending=False)
)
delivery_return["return_rate"] = (delivery_return["mean"] * 100)

                # NUMERICAL FEATURES BY RETURN STATUS
numerical_features = ["price","discount","quantity","total_amount","shipping_cost","profit_margin","customer_age","delivery_days"]

numerical_comparison = ( df.groupby("returned")[numerical_features].mean())

                # CORRELATION WITH TARGET
correlation = (df[numerical_features + ["returned"]].corr()["returned"]
.sort_values(ascending=False))
print(correlation)

                # CREATE SIMPLE VISUALIZATIONS
category_return["return_rate"].plot(kind="bar",figsize=(8, 5))
plt.title("Return Rate by Category")
plt.xlabel("Category")
plt.ylabel("Return Rate (%)")
plt.tight_layout()
plt.show()

region_return["return_rate"].plot(kind="bar",figsize=(8, 5))
plt.title("Return Rate by Region")
plt.xlabel("Region")
plt.ylabel("Return Rate (%)")
plt.tight_layout()
plt.show()

discount_return["return_rate"].plot(kind="bar",figsize=(8, 5))
plt.title("Return Rate by Discount")
plt.xlabel("Discount")
plt.ylabel("Return Rate (%)")
plt.tight_layout()
plt.show()