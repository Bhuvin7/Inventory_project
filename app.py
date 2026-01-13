import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import plotly.express as px

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AI Inventory Dashboard", layout="wide")
st.title("📦 AI-Driven Demand Forecasting & Inventory Optimization")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload Sales Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # -------------------------------
    # Load & Process Data
    # -------------------------------
    df = pd.read_csv(uploaded_file)

    # Make sure columns exist
    expected_columns = ['Category','Region','Price','Discount','Weather Condition','Promotion','Seasonality','Actual_Demand']
    if not all(col in df.columns for col in expected_columns):
        st.error(f"CSV is missing some required columns. Required: {expected_columns}")
    else:
        # Feature Engineering
        df['Sales_Lag_7'] = df.groupby('Category')['Actual_Demand'].shift(7)
        df['Sales_Lag_30'] = df.groupby('Category')['Actual_Demand'].shift(30)
        df['Rolling_Mean_7'] = df.groupby('Category')['Actual_Demand'].rolling(7).mean().reset_index(0, drop=True)

        # Label Encoding
        le = LabelEncoder()
        for col in ['Category', 'Region', 'Weather Condition', 'Seasonality']:
            df[col] = le.fit_transform(df[col])
        df.dropna(inplace=True)

        # Features & Target
        X = df[['Category','Region','Price','Discount','Weather Condition','Promotion','Seasonality',
                'Sales_Lag_7','Sales_Lag_30','Rolling_Mean_7']]
        y = df['Actual_Demand']

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Model Training
        model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # -------------------------------
        # Inventory Optimization
        # -------------------------------
        lead_time = st.slider("Lead Time (days)", 1, 10, 3)
        safety_stock = 1.65 * rmse

        results = X_test.copy()
        results['Actual_Demand'] = y_test.values
        results['Predicted_Demand'] = y_pred
        results['Reorder_Point'] = (results['Predicted_Demand'] * lead_time + safety_stock).round()
        results['Suggested_Order'] = results['Reorder_Point']

        # -------------------------------
        # Display Table
        # -------------------------------
        st.subheader("📊 Inventory Recommendation Table")
        st.dataframe(results[['Actual_Demand','Predicted_Demand','Reorder_Point','Suggested_Order']].head(20))

        # -------------------------------
        # Download Button
        # -------------------------------
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="inventory_recommendations.csv",
            mime="text/csv"
        )

        # -------------------------------
        # Charts
        # -------------------------------

        # 1. Predicted vs Actual Demand Line Chart
        st.subheader("📈 Predicted vs Actual Demand")
        fig1 = px.line(results, y=['Actual_Demand', 'Predicted_Demand'], title='Actual vs Predicted Demand')
        st.plotly_chart(fig1, use_container_width=True)

        # 2. Pie Chart: Stock Distribution
        st.subheader("🥧 Inventory Stock Distribution")
        total_actual = results['Actual_Demand'].sum()
        total_predicted = results['Predicted_Demand'].sum()
        total_suggested = results['Suggested_Order'].sum()

        stock_summary = pd.DataFrame({
            'Status': ['Current Demand', 'Predicted Demand', 'Suggested Order'],
            'Units': [total_actual, total_predicted, total_suggested]
        })
        fig2 = px.pie(stock_summary, names='Status', values='Units', title='Stock Distribution')
        st.plotly_chart(fig2, use_container_width=True)

        # 3. Bar Chart: Top 10 Products by Reorder Point
        st.subheader("📊 Top 10 Products to Reorder")
        top_products = results.sort_values(by='Reorder_Point', ascending=False).head(10)
        fig3 = px.bar(top_products, x=top_products.index, y='Reorder_Point', color='Suggested_Order',
                      title='Top 10 Products to Reorder', labels={'x':'Record Index','Reorder_Point':'Reorder Point'})
        st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("Please upload your sales dataset to begin.")
