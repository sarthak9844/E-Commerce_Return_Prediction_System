import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="E-commerce Return Prediction",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f5f7fb;
    }

    /* Remove top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    /* Hero section */
    .hero {
        padding: 30px 35px;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1f2937 55%,
            #374151 100%
        );
        color: white;
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #d1d5db;
        margin-bottom: 0;
    }

    /* Section headings */
    .section-title {
        font-size: 23px;
        font-weight: 700;
        color: #111827;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    /* Cards */
    .info-card {
        background-color: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        margin-bottom: 15px;
    }

    .metric-title {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 5px;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #111827;
    }

    /* Result cards */
    .return-card {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        padding: 25px;
        border-radius: 16px;
        margin-top: 20px;
    }

    .not-return-card {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        padding: 25px;
        border-radius: 16px;
        margin-top: 20px;
    }

    .result-title {
        font-size: 25px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .result-text {
        font-size: 16px;
        color: #374151;
    }

    /* Prediction probability */
    .probability-number {
        font-size: 42px;
        font-weight: 800;
        color: #111827;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 10px;
        font-size: 17px;
        font-weight: 700;
        border: none;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        padding-top: 35px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(
    r"D:\PROJECT\E-commerce _Return_prediction_system\model\logistic_regression_model.pkl"
)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def create_feature(df):

    df = df.copy()

    df["price_per_quantity"] = (
        df["price"] / df["quantity"].replace(0, 1)
    )

    df["amount_per_quantity"] = (
        df["total_amount"] / df["quantity"].replace(0, 1)
    )

    df["shipping_ratio"] = (
        df["shipping_cost"] /
        df["total_amount"].replace(0, 1)
    )

    df["profit_amount"] = (
        df["total_amount"] *
        df["profit_margin"] / 100
    )

    df["age_group"] = pd.cut(
        df["customer_age"],
        bins=[0, 25, 35, 50, 65, 100],
        labels=[
            "18-25",
            "26-35",
            "36-50",
            "51-65",
            "66+"
        ]
    )

    df["delivery_group"] = pd.cut(
        df["delivery_days"],
        bins=[
            -1,
            2,
            5,
            10,
            float("inf")
        ],
        labels=[
            "Fast",
            "Normal",
            "Slow",
            "Very_Slow"
        ]
    )

    return df


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 📦E-commerece Return Predictor")

    st.markdown("---")

    st.markdown("### About")

    st.write(
        """
        This machine learning application predicts
        whether an e-commerce order is likely to be
        returned based on order, customer and delivery
        information.
        """
    )

    st.markdown("---")

    st.markdown("### 🤖 Model")

    st.write("**Algorithm:** Random Forest")
    st.write("**Task:** Binary Classification")
    st.write("**Target:** Returned")
    st.write("**Preprocessing:** Scaling + One-Hot Encoding")

    st.markdown("---")

    st.markdown("### 📊 Model Performance")

    
    st.metric(
        "Return Recall",
        "68%"
    )

    st.markdown("---")

    st.caption(
        "E-commerce Return Prediction System"
    )


# =========================================================
# HERO HEADER
# =========================================================

st.markdown("""
<div class="hero">
<div class="hero-title">📦 E-commerce Return Prediction</div>
<div class="hero-subtitle">Machine Learning powered prediction systemfor identifying potentially returned orders.</div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# DASHBOARD SUMMARY
# =========================================================

st.markdown(
    '<div class="section-title">📊 Prediction Dashboard</div>',
    unsafe_allow_html=True
)

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.markdown("""
    <div class="info-card">
        <div class="metric-title">Model</div>
        <div class="metric-value">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

with metric2:
    st.markdown("""
    <div class="info-card">
        <div class="metric-title">Prediction Type</div>
        <div class="metric-value">Binary</div>
    </div>
    """, unsafe_allow_html=True)

with metric3:
    st.markdown("""
    <div class="info-card">
        <div class="metric-title">Target</div>
        <div class="metric-value">Returned</div>
    </div>
    """, unsafe_allow_html=True)

with metric4:
    st.markdown("""
    <div class="info-card">
        <div class="metric-title">Decision Threshold</div>
        <div class="metric-value">43%</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown(
    '<div class="section-title">🛒 Order Information</div>',
    unsafe_allow_html=True
)

st.caption(
    "Enter the details of an order to generate a return prediction."
)


left, middle, right = st.columns(3)


# ---------------------------------------------------------
# COLUMN 1
# ---------------------------------------------------------

with left:

    st.markdown("#### 📦 Product")

    category = st.selectbox(
        "Category",
        [
            "Fashion",
            "Electronics",
            "Home",
            "Toys",
            "Sports",
            "Beauty",
            "Grocery"
        ]
    )

    price = st.number_input(
        "Price",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

    quantity = st.number_input(
        "Quantity",
        min_value=1,
        value=1,
        step=1
    )

    discount = st.number_input(
        "Discount",
        min_value=0.0,
        value=0.15,
        step=0.01,
        format="%.2f",
        help="Use the same format as the training dataset, e.g. 0.15."
    )


# ---------------------------------------------------------
# COLUMN 2
# ---------------------------------------------------------

with middle:

    st.markdown("#### 💳 Order & Payment")

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Credit Card",
            "Debit Card",
            "UPI",
            "Cash on Delivery",
            "Net Banking"
        ]
    )

    region = st.selectbox(
        "Region",
        [
            "North",
            "South",
            "East",
            "West",
            "Central"
        ]
    )

    total_amount = st.number_input(
        "Total Amount",
        min_value=0.0,
        value=1000.0,
        step=50.0
    )

    shipping_cost = st.number_input(
        "Shipping Cost",
        min_value=0.0,
        value=50.0,
        step=5.0
    )

    profit_margin = st.number_input(
        "Profit Margin",
        min_value=-100.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )


# ---------------------------------------------------------
# COLUMN 3
# ---------------------------------------------------------

with right:

    st.markdown("#### 👤 Customer & Delivery")

    customer_age = st.number_input(
        "Customer Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

    customer_gender = st.selectbox(
        "Customer Gender",
        [
            "Male",
            "Female",
            "Other"
        ]
    )

    delivery_days = st.number_input(
        "Delivery Days",
        min_value=0,
        value=3,
        step=1
    )

    order_year = st.number_input(
        "Order Year",
        min_value=2020,
        max_value=2030,
        value=2025,
        step=1
    )

    order_month = st.number_input(
        "Order Month",
        min_value=1,
        max_value=12,
        value=6,
        step=1
    )

    order_day_of_week = st.number_input(
        "Order Day of Week",
        min_value=0,
        max_value=6,
        value=2,
        step=1,
        help="0 = Monday, 6 = Sunday"
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("---")

predict_col, _ = st.columns([1, 2])

with predict_col:

    predict_button = st.button(
        "🔍 Predict Return",
        type="primary"
    )


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # Create input dataframe

    input_data = pd.DataFrame({

        "category": [category],

        "price": [price],

        "discount": [discount],

        "quantity": [quantity],

        "payment_method": [payment_method],

        "region": [region],

        "total_amount": [total_amount],

        "shipping_cost": [shipping_cost],

        "profit_margin": [profit_margin],

        "customer_age": [customer_age],

        "customer_gender": [customer_gender],

        "delivery_days": [delivery_days],

        "order_year": [order_year],

        "order_month": [order_month],

        "order_day_of_week": [order_day_of_week]
    })


    # Feature engineering

    input_data = create_feature(input_data)


    # Prediction

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown(
        '<div class="section-title">🎯 Prediction Result</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = st.columns([1.4, 1])


    # -----------------------------------------------------
    # RESULT MESSAGE
    # -----------------------------------------------------

    with result_col1:

        if prediction == 1:

            st.markdown(f"""
            <div class="return-card">
            <div class="result-title">⚠️ Return Likely</div>
            <div class="result-text">The model predicts that this order is likely to be returned.</div>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="not-return-card">
            <div class="result-title">✅ Return Unlikely</div>
            <div class="result-text">The model predicts that this order is unlikely to be returned.</div>
            </div>
            """, unsafe_allow_html=True)


    # -----------------------------------------------------
    # PROBABILITY
    # -----------------------------------------------------

    with result_col2:

        st.markdown("""
        <div class="info-card">
        <div class="metric-title">Return Probability</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
             f'<div class="probability-number">' 
             f'{probability:.2%}' f'</div>', unsafe_allow_html=True
        )

        st.progress(
            float(probability)
        )

        if probability >= 0.5:

            st.caption(
                "Probability is above the 38% decision threshold."
            )

        else:

            st.caption(
                "Probability is below the 38% decision threshold."
            )


    # =====================================================
    # ORDER SUMMARY
    # =====================================================

    st.markdown(
        '<div class="section-title">📋 Order Summary</div>',
        unsafe_allow_html=True
    )

    summary1, summary2, summary3, summary4 = st.columns(4)

    with summary1:
        st.metric(
            "Category",
            category
        )

    with summary2:
        st.metric(
            "Price",
            f"₹{price:,.2f}"
        )

    with summary3:
        st.metric(
            "Delivery",
            f"{delivery_days} days"
        )

    with summary4:
        st.metric(
            "Customer Age",
            f"{customer_age}"
        )


# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🧠 How It Works</div>',
    unsafe_allow_html=True
)

how1, how2, how3 = st.columns(3)

with how1:

    st.markdown("""
    <div class="info-card">

    ### 1. Enter Order Data

    Provide product, payment, customer
    and delivery information.

    </div>
    """, unsafe_allow_html=True)


with how2:

    st.markdown("""
    <div class="info-card">

    ### 2. Feature Engineering

    The application creates additional
    features such as shipping ratio,
    profit amount and customer age group.

    </div>
    """, unsafe_allow_html=True)


with how3:

    st.markdown("""
    <div class="info-card">


    Random Forest calculates the
    probability of the order being returned.

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

    E-commerce Return Prediction System
    • Machine Learning Project
    • Built with Python & Streamlit

</div>
""", unsafe_allow_html=True)
