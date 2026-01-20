import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI-Driven Inventory Optimization",
    layout="wide"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("📦 AI-Driven Inventory Optimization System")
st.write(
    "Upload your sales dataset to analyze demand, forecast inventory, "
    "and receive stock recommendations."
)

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# --------------------------------------------------
# DATASET REQUIREMENTS
# --------------------------------------------------
st.markdown("### 📄 Required CSV Columns")
st.markdown("""
Your dataset **must contain** the following columns:

- **Category** – Product category (encoded or text)
- **Region** – Sales region
- **Price** – Product price
- **Discount** – Discount percentage
- **Weather Condition** – Weather indicator
- **Promotion** – 0 = No, 1 = Yes
- **Seasonality** – Seasonal indicator
- **Sales_Lag_7** – Sales 7 days ago
- **Sales_Lag_30** – Sales 30 days ago
- **Rolling_Mean_7** – 7-day rolling mean
- **Actual_Demand** – Real demand
- **Predicted_Demand** – Model prediction
- **Reorder_Point** – Inventory reorder level
- **Suggested_Order** – Recommended order quantity
""")

# --------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("✅ Dataset loaded successfully!")

    # --------------------------------------------------
    # BASIC METRICS
    # --------------------------------------------------
    rmse = np.sqrt(
        np.mean((df["Actual_Demand"] - df["Predicted_Demand"]) ** 2)
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Sales", int(df["Actual_Demand"].sum()))
    col2.metric("Avg Predicted Demand", int(df["Predicted_Demand"].mean()))
    col3.metric("RMSE", round(rmse, 2))
    col4.metric(
        "Inventory Alert %",
        f"{round((df['Suggested_Order'] > df['Actual_Demand']).mean()*100,1)}%"
    )

    st.divider()

    # --------------------------------------------------
    # ACTUAL VS PREDICTED BAR CHART
    # --------------------------------------------------
    st.subheader("📊 Actual vs Predicted Demand")
    st.bar_chart(
        df[["Actual_Demand", "Predicted_Demand"]].head(20)
    )

    # --------------------------------------------------
    # DEMAND TREND LINE CHART
    # --------------------------------------------------
    st.subheader("📈 Demand Trend")
    st.line_chart(
        df[["Actual_Demand", "Predicted_Demand"]]
    )

    # --------------------------------------------------
    # INVENTORY DISTRIBUTION PIE CHART
    # --------------------------------------------------
    st.subheader("🥧 Inventory Distribution")

    pie_data = df[["Actual_Demand", "Suggested_Order"]].sum()

    fig, ax = plt.subplots()
    ax.pie(
        pie_data,
        labels=pie_data.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax.axis("equal")
    st.pyplot(fig)

    st.divider()

    # --------------------------------------------------
    # CATEGORY-WISE ANALYSIS
    # --------------------------------------------------
    st.subheader("🔍 Category-wise Demand Analysis")

    selected_category = st.selectbox(
        "Select Category",
        df["Category"].unique()
    )

    filtered_df = df[df["Category"] == selected_category]

    st.bar_chart(
        filtered_df[["Actual_Demand", "Predicted_Demand"]]
    )

    # --------------------------------------------------
    # LOW STOCK ALERT
    # --------------------------------------------------
    st.subheader("⚠️ Low Stock Products")

    low_stock = df[df["Suggested_Order"] > df["Actual_Demand"]]

    st.dataframe(
        low_stock[
            [
                "Category",
                "Actual_Demand",
                "Predicted_Demand",
                "Reorder_Point",
                "Suggested_Order"
            ]
        ].head(10)
    )

    # --------------------------------------------------
    # FULL INVENTORY TABLE
    # --------------------------------------------------
    st.subheader("📋 Inventory Recommendation Table")
    st.dataframe(df.head(20))

    # --------------------------------------------------
    # DOWNLOAD RESULTS
    # --------------------------------------------------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Results as CSV",
        csv,
        "inventory_recommendations.csv",
        "text/csv"
    )

else:
    st.info("📂 Please upload a CSV file to begin analysis.")

