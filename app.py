import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Inventory Dashboard", layout="wide")
st.title("📦 AI-Driven Inventory Optimization System")

st.markdown("""
Upload your sales dataset to analyze **category-wise demand**,  
predict **future demand**, and get **inventory recommendations**.
""")
st.subheader("📄 Dataset Requirements")

st.markdown("""
Your CSV file **must contain the following columns**:

- **Category** → Product category (Electronics / Clothing / Groceries)
- **Region** → Region or store ID
- **Price** → Product price
- **Discount** → Discount percentage
- **Weather Condition** → Encoded weather value
- **Promotion** → 0 = No promotion, 1 = Promotion
- **Seasonality** → Seasonal indicator
- **Sales_Lag_7** → Sales 7 days ago
- **Sales_Lag_30** → Sales 30 days ago
- **Rolling_Mean_7** → 7-day rolling average sales
- **Actual_Demand** → Actual demand value

⚠️ **File must be in CSV format (.csv)**
""")

uploaded_file = st.file_uploader(
    "📤 Upload your sales dataset",
    type=["csv"]
)


# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # ---------------- COLUMN CHECK ----------------
    required_columns = [
        'Category', 'Region', 'Price', 'Discount',
        'Weather Condition', 'Promotion', 'Seasonality',
        'Actual_Demand'
    ]

    if not all(col in df.columns for col in required_columns):
        st.error("❌ Dataset does not match required format.")
        st.stop()

    # ---------------- FEATURE ENGINEERING ----------------
    df['Sales_Lag_7'] = df.groupby('Category')['Actual_Demand'].shift(7)
    df['Sales_Lag_30'] = df.groupby('Category')['Actual_Demand'].shift(30)
    df['Rolling_Mean_7'] = (
        df.groupby('Category')['Actual_Demand']
        .rolling(7)
        .mean()
        .reset_index(0, drop=True)
    )

    df.dropna(inplace=True)

    # ---------------- ENCODING ----------------
    encoder = LabelEncoder()
    for col in ['Category', 'Region', 'Weather Condition', 'Seasonality']:
        df[col] = encoder.fit_transform(df[col])

    # ---------------- SIDEBAR FILTER ----------------
    st.sidebar.header("🔎 Filter")
    selected_category = st.sidebar.selectbox(
        "Select Category", sorted(df['Category'].unique())
    )

    filtered_df = df[df['Category'] == selected_category]

    # ---------------- MODEL DATA ----------------
    X = filtered_df[
        ['Category', 'Region', 'Price', 'Discount',
         'Weather Condition', 'Promotion', 'Seasonality',
         'Sales_Lag_7', 'Sales_Lag_30', 'Rolling_Mean_7']
    ]

    y = filtered_df['Actual_Demand']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # ---------------- MODEL ----------------
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # ---------------- INVENTORY LOGIC ----------------
    lead_time = st.sidebar.slider("Lead Time (Days)", 1, 10, 3)
    safety_stock = 1.65 * rmse

    results = X_test.copy()
    results['Actual_Demand'] = y_test.values
    results['Predicted_Demand'] = y_pred
    results['Reorder_Point'] = (
        results['Predicted_Demand'] * lead_time + safety_stock
    ).round()
    results['Suggested_Order'] = results['Reorder_Point']

    # ---------------- METRICS ----------------
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", round(rmse, 2))
    col2.metric("Avg Predicted Demand", int(results['Predicted_Demand'].mean()))
    col3.metric("Avg Suggested Order", int(results['Suggested_Order'].mean()))

    # ---------------- TABLE ----------------
    st.subheader("📋 Inventory Recommendation Table")
    st.dataframe(
        results[['Actual_Demand', 'Predicted_Demand',
                 'Reorder_Point', 'Suggested_Order']]
    )

    # ---------------- CHARTS ----------------

    # Line Chart
    st.subheader("📈 Actual vs Predicted Demand")
    fig1 = px.line(
        results,
        y=['Actual_Demand', 'Predicted_Demand'],
        title="Demand Comparison"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Pie Chart
    st.subheader("🥧 Inventory Distribution")
    pie_df = pd.DataFrame({
        "Type": ["Actual Demand", "Predicted Demand", "Suggested Order"],
        "Units": [
            results['Actual_Demand'].sum(),
            results['Predicted_Demand'].sum(),
            results['Suggested_Order'].sum()
        ]
    })

    fig2 = px.pie(pie_df, names='Type', values='Units')
    st.plotly_chart(fig2, use_container_width=True)

    # Bar Chart
    st.subheader("📊 Top Inventory Requirements")
    top_items = results.sort_values(
        by='Reorder_Point', ascending=False
    ).head(10)

    fig3 = px.bar(
        top_items,
        y='Reorder_Point',
        title="Top Items to Reorder"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ---------------- DOWNLOAD ----------------
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Inventory Report",
        csv,
        "inventory_output.csv",
        "text/csv"
    )

else:
    st.info("📂 Please upload a CSV file to start analysis.")


