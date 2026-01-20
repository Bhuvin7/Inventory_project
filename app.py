import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="AI Inventory Optimization Dashboard",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("📦 AI-Driven Inventory Optimization System")
st.write(
    "Upload your sales dataset to analyze demand, forecast inventory, "
    "and get stock recommendations."
)

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Drag & drop your CSV file here",
    type=["csv"]
)

# --------------------------------------------------
# REQUIRED COLUMNS INFO
# --------------------------------------------------
st.markdown("### 📄 Required Dataset Columns")
st.markdown("""
Your dataset should contain **most of the following columns**:

- Category  
- Region  
- Price  
- Discount  
- Weather Condition  
- Promotion  
- Seasonality  
- Sales_Lag_7  
- Sales_Lag_30  
- Rolling_Mean_7  
- Actual_Demand / Demand / Units Sold  
- Predicted_Demand / Forecast  
- Reorder_Point  
- Suggested_Order  
""")

# --------------------------------------------------
# HELPER FUNCTION FOR COLUMN MATCHING
# --------------------------------------------------
def find_column(possible_names, columns):
    for name in possible_names:
        if name in columns:
            return name
    return None

# --------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("✅ Dataset uploaded successfully")

    # Show detected columns
    st.subheader("🧾 Detected Columns")
    st.write(list(df.columns))

    # Identify key columns safely
    actual_col = find_column(
        ["Actual_Demand", "Actual Demand", "Demand", "Units Sold"],
        df.columns
    )

    pred_col = find_column(
        ["Predicted_Demand", "Predicted Demand", "Forecast"],
        df.columns
    )

    reorder_col = find_column(
        ["Reorder_Point", "Reorder Point"],
        df.columns
    )

    suggest_col = find_column(
        ["Suggested_Order", "Suggested Order"],
        df.columns
    )

    category_col = find_column(
        ["Category"],
        df.columns
    )

    # Stop if critical columns missing
    if actual_col is None or pred_col is None:
        st.error("❌ Actual or Predicted demand column not found.")
        st.stop()

    # --------------------------------------------------
    # KPI METRICS
    # --------------------------------------------------
    rmse = np.sqrt(np.mean((df[actual_col] - df[pred_col]) ** 2))

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Actual Demand", int(df[actual_col].sum()))
    col2.metric("Avg Predicted Demand", int(df[pred_col].mean()))
    col3.metric("RMSE", round(rmse, 2))

    if suggest_col:
        alert_pct = (df[suggest_col] > df[actual_col]).mean() * 100
        col4.metric("Inventory Alert %", f"{round(alert_pct, 1)}%")
    else:
        col4.metric("Inventory Alert %", "N/A")

    st.divider()

    # --------------------------------------------------
    # ACTUAL VS PREDICTED BAR CHART
    # --------------------------------------------------
    st.subheader("📊 Actual vs Predicted Demand")
    st.bar_chart(df[[actual_col, pred_col]].head(20))

    # --------------------------------------------------
    # DEMAND TREND LINE CHART
    # --------------------------------------------------
    st.subheader("📈 Demand Trend")
    st.line_chart(df[[actual_col, pred_col]])

    st.divider()

    # --------------------------------------------------
    # CATEGORY-WISE ANALYSIS
    # --------------------------------------------------
    if category_col:
        st.subheader("🔍 Category-wise Demand Analysis")

        selected_category = st.selectbox(
            "Select Category",
            df[category_col].unique()
        )

        filtered_df = df[df[category_col] == selected_category]

        st.bar_chart(
            filtered_df[[actual_col, pred_col]]
        )

    # --------------------------------------------------
    # LOW STOCK / INVENTORY TABLE
    # --------------------------------------------------
    if suggest_col:
        st.subheader("⚠️ Inventory Recommendation Table")

        display_cols = [
            col for col in
            [category_col, actual_col, pred_col, reorder_col, suggest_col]
            if col is not None
        ]

        st.dataframe(df[display_cols].head(20))

    # --------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------
    st.download_button(
        "⬇️ Download Results as CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="inventory_results.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Upload a CSV file to start the analysis.")
