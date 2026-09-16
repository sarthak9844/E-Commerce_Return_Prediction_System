import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(
    r"D:\PROJECT\E-commerce _Return_prediction_system\Dataset\Cleaned_Ecommerce_Data.csv"
)

# Count returned and not returned orders
return_counts = df["returned"].value_counts().sort_index()

labels = ["Not Returned", "Returned"]
counts = [
    return_counts.get(0, 0),
    return_counts.get(1, 0)
]

# Plot
plt.figure(figsize=(8, 5))

bars = plt.bar(labels, counts)

plt.title("E-commerce Order Return Distribution")
plt.xlabel("Return Status")
plt.ylabel("Number of Orders")

# Add count and percentage
total = sum(counts)

for bar, count in zip(bars, counts):
    percentage = (count / total) * 100

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{count:,}\n({percentage:.2f}%)",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()
plt.savefig(
    r"D:\PROJECT\E-commerce _Return_prediction_system\notebook\images\class_imbalance.png",
    dpi=300,
    bbox_inches="tight"
)

