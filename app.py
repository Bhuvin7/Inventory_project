import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Demand Forecasting", layout="wide")

st.title("📊 AI-Based Demand Forecasting Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your sales dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- DATE FIX ---
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['Date'])

    # --- KPI CALCULATIONS ---
    total_sales = df['Units Sold'].sum()
    total_demand = df['Demand'].sum()
    avg_price = df['Price'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Units Sold", f"{total_sales:,}")
    col2.metric("Total Predicted Demand", f"{total_demand:,}")
    col3.metric("Average Price", f"₹ {avg_price:,.2f}")

    # --- TIME SERIES CHART ---
    df = df.sort_values('Date')
    monthly = df.groupby(df['Date'].dt.to_period("M")).agg({
        'Units Sold': 'sum',
        'Demand': 'sum'
    }).reset_index()

    monthly['Date'] = monthly['Date'].astype(str)

    fig = px.line(monthly, x='Date', y=['Units Sold', 'Demand'],
                  title="Sales vs Predicted Demand Over Time")

    st.plotly_chart(fig, use_container_width=True)

    # --- OUTPUT TABLE ---
    st.subheader("📋 Forecast Output Data")
    st.dataframe(monthly)

else:
    st.info("Please upload a dataset to see the dashboard.")
