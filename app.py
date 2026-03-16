import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Inventory System",
    layout="wide",
    page_icon="📦",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #1d4ed8 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricLabel"] { font-size: 13px !important; color: #6b7280 !important; }
</style>
""", unsafe_allow_html=True)


# ── TITLE ─────────────────────────────────────────────────────────────────────
st.title("📦 AI-Driven Inventory Optimization System")
st.markdown("Upload your sales dataset to get **demand forecasts**, **inventory metrics**, and a **downloadable report**.")

# ── FILE UPLOAD ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file is None:
    st.info("👆 Upload a CSV file to get started. Expected columns: Date, Units Sold, Inventory Level, Price, Product ID, Category.")
    st.stop()


# ── LOAD & CLEAN DATA ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_process(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])

    # Numeric columns
    df["Units Sold"]      = pd.to_numeric(df.get("Units Sold"),      errors="coerce").fillna(0).astype(int)
    df["Inventory Level"] = pd.to_numeric(df.get("Inventory Level"), errors="coerce").fillna(0)
    df["Price"]           = pd.to_numeric(df.get("Price"),           errors="coerce").fillna(0)

    # Daily aggregation
    daily = df.groupby("Date", as_index=False).agg({
        "Units Sold":      "sum",
        "Inventory Level": "sum",
        "Price":           "mean",
    }).sort_values("Date").reset_index(drop=True)

    # Feature engineering
    daily["Lag_7"]           = daily["Units Sold"].shift(7)
    daily["Lag_30"]          = daily["Units Sold"].shift(30)
    daily["Rolling_Mean_7"]  = daily["Units Sold"].shift(1).rolling(7).mean()
    daily["Rolling_Mean_30"] = daily["Units Sold"].shift(1).rolling(30).mean()

    clean = daily.dropna().reset_index(drop=True)

    # Train model
    features = ["Lag_7", "Lag_30", "Rolling_Mean_7", "Rolling_Mean_30"]
    X = clean[features]
    y = clean["Units Sold"]
    split = int(len(X) * 0.85)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X[:split], y[:split])

    preds     = model.predict(X)
    rmse      = float(np.sqrt(mean_squared_error(y[split:], model.predict(X[split:]))))
    accuracy  = float(max(0, 100 - (rmse / max(y.mean(), 1) * 100)))

    # Safety stock (Z=1.65, lead time=7 days)
    sigma        = float(daily["Units Sold"].std())
    safety_stock = int(round(1.65 * sigma * np.sqrt(7)))
    avg_demand   = float(daily["Units Sold"].mean())
    reorder_pt   = int(round(avg_demand * 7 + safety_stock))

    # Build downloadable report
    report = clean[["Date", "Units Sold", "Lag_7", "Lag_30", "Rolling_Mean_7", "Rolling_Mean_30"]].copy()
    report["Predicted_Demand"] = preds.round(0).astype(int)
    report["Safety_Stock"]     = safety_stock
    report["Reorder_Point"]    = reorder_pt
    report.columns = [
        "Date", "Current_Demand", "Lag_7", "Lag_30",
        "Rolling_Mean_7", "Rolling_Mean_30",
        "Predicted_Demand", "Safety_Stock", "Reorder_Point"
    ]
    report["Date"] = report["Date"].dt.strftime("%Y-%m-%d")

    return daily, clean, report, rmse, accuracy, safety_stock, reorder_pt, avg_demand


file_bytes = uploaded_file.read()
daily, clean, report, rmse, accuracy, safety_stock, reorder_pt, avg_demand = load_and_process(file_bytes)


# ── KPI METRICS ───────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Total Units Sold",    f"{int(daily['Units Sold'].sum()):,}")
with c2: st.metric("Avg Daily Demand",    f"{int(avg_demand):,}")
with c3: st.metric("Safety Stock",        f"{safety_stock:,}")
with c4: st.metric("Reorder Point",       f"{reorder_pt:,}")
with c5: st.metric("Forecast Accuracy",   f"{accuracy:.1f}%", f"RMSE: {rmse:.0f}")
st.markdown("---")


# ── CHART 1: Actual vs Predicted ──────────────────────────────────────────────
st.subheader("📈 Actual vs Predicted Demand")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=clean["Date"], y=clean["Units Sold"],
    name="Actual Demand", mode="lines",
    line=dict(color="#1d4ed8", width=2),
))
fig1.add_trace(go.Scatter(
    x=clean["Date"], y=report["Predicted_Demand"].values,
    name="Predicted Demand", mode="lines",
    line=dict(color="#f97316", width=1.5, dash="dot"),
))
fig1.update_layout(
    template="plotly_white",
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=10, b=0), height=320,
)
st.plotly_chart(fig1, use_container_width=True)


# ── CHART 2: Rolling Mean & Inventory Level ───────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Rolling Mean (7-day & 30-day)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=clean["Date"], y=clean["Rolling_Mean_7"],
        name="7-Day Rolling Mean", mode="lines",
        line=dict(color="#7c3aed", width=2),
    ))
    fig2.add_trace(go.Scatter(
        x=clean["Date"], y=clean["Rolling_Mean_30"],
        name="30-Day Rolling Mean", mode="lines",
        line=dict(color="#059669", width=2),
    ))
    fig2.update_layout(
        template="plotly_white",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=10, b=0), height=280,
    )
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("🏭 Inventory Level Over Time")
    fig3 = px.area(
        daily, x="Date", y="Inventory Level",
        color_discrete_sequence=["#0ea5e9"],
        template="plotly_white",
    )
    fig3.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        margin=dict(l=0, r=0, t=10, b=0), height=280,
    )
    st.plotly_chart(fig3, use_container_width=True)


# ── CHART 3: Lag 7 & Lag 30 ───────────────────────────────────────────────────
st.subheader("🔁 Lag Features (Lag 7 & Lag 30)")
fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    x=clean["Date"], y=clean["Lag_7"],
    name="Lag 7", mode="lines",
    line=dict(color="#db2777", width=1.5),
))
fig4.add_trace(go.Scatter(
    x=clean["Date"], y=clean["Lag_30"],
    name="Lag 30", mode="lines",
    line=dict(color="#d97706", width=1.5),
))
fig4.update_layout(
    template="plotly_white",
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=10, b=0), height=280,
)
st.plotly_chart(fig4, use_container_width=True)


# ── REPORT TABLE + DOWNLOAD ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Inventory Recommendation Table")
st.dataframe(report.tail(30), use_container_width=True, hide_index=True)

csv_bytes = report.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Full Report (inventory_recommendations.csv)",
    data=csv_bytes,
    file_name="inventory_recommendations.csv",
    mime="text/csv",
    use_container_width=True,
)
