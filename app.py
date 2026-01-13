if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(by=['Product ID', 'Date'])

    # Feature engineering
    df['Sales_Lag_7'] = df.groupby('Product ID')['Units Sold'].shift(7)
    df['Sales_Lag_30'] = df.groupby('Product ID')['Units Sold'].shift(30)
    df['Rolling_Mean_7'] = df.groupby('Product ID')['Units Sold'].rolling(7).mean().reset_index(0, drop=True)

    # Label encoding
    le = LabelEncoder()
    for col in ['Category', 'Region', 'Weather Condition', 'Seasonality']:
        df[col] = le.fit_transform(df[col])
    df.dropna(inplace=True)

    # Features and target
    X = df[['Category','Region','Price','Discount','Weather Condition','Promotion','Seasonality','Sales_Lag_7','Sales_Lag_30','Rolling_Mean_7']]
    y = df['Demand']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Inventory Optimization
    lead_time = st.slider("Lead Time (days)", 1, 10, 3)
    safety_stock = 1.65 * rmse

    results = X_test.copy()
    results['Actual_Demand'] = y_test.values
    results['Predicted_Demand'] = y_pred
    results['Reorder_Point'] = (results['Predicted_Demand'] * lead_time + safety_stock).round()
    results['Suggested_Order'] = results['Reorder_Point']

    st.subheader("📊 Inventory Recommendation Table")
    st.dataframe(results[['Actual_Demand','Predicted_Demand','Reorder_Point','Suggested_Order']].head(20))
    st.success("✅ Inventory recommendations generated successfully!")

    # ✅ ADD DOWNLOAD BUTTON INSIDE THIS BLOCK
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="inventory_recommendations.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload your sales dataset to begin.")
