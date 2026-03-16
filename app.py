import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Inventory System", layout="wide", page_icon="📦")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stMetricValue"] {
    font-size: 26px !important; font-weight: 700 !important;
    color: #1d4ed8 !important; font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricLabel"] { font-size: 13px !important; color: #6b7280 !important; }
</style>
""", unsafe_allow_html=True)


# ── TITLE ─────────────────────────────────────────────────────────────────────
st.title("📦 AI-Driven Inventory Optimization System")
st.markdown("Upload your sales dataset to analyze **product-wise demand**, predict **future demand**, and get **inventory recommendations**.")

uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file is None:
    st.info("👆 Upload a CSV file to get started.")
    st.stop()


# ── LOAD & PROCESS ────────────────────────────────────────────────────────────
@st.cache_data
def load_and_process(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")
    df["Date"]            = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Units Sold"]      = pd.to_numeric(df.get("Units Sold"),      errors="coerce").fillna(0).astype(int)
    df["Inventory Level"] = pd.to_numeric(df.get("Inventory Level"), errors="coerce").fillna(0)
    df["Price"]           = pd.to_numeric(df.get("Price"),           errors="coerce").fillna(0)
    df = df.dropna(subset=["Date"])
    return df

file_bytes = uploaded_file.read()
raw = load_and_process(file_bytes)

# ── PRODUCT SELECTOR ──────────────────────────────────────────────────────────
all_products = sorted(raw["Product ID"].unique())
selected_product = st.selectbox("🔍 Select Product ID to Analyse", all_products)

# Filter to selected product
prod_df = raw[raw["Product ID"] == selected_product].copy()
category = prod_df["Category"].iloc[0] if "Category" in prod_df.columns else "—"

# Daily aggregate for selected product
daily = prod_df.groupby("Date", as_index=False).agg({
    "Units Sold":      "sum",
    "Inventory Level": "sum",
    "Price":           "mean",
}).sort_values("Date").reset_index(drop=True)


# ── FORECAST FOR SELECTED PRODUCT ─────────────────────────────────────────────
@st.cache_data
def forecast_product(product_id, file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")
    df["Date"]       = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Units Sold"] = pd.to_numeric(df["Units Sold"], errors="coerce").fillna(0).astype(int)
    df["Inventory Level"] = pd.to_numeric(df.get("Inventory Level"), errors="coerce").fillna(0)
    df["Price"]      = pd.to_numeric(df.get("Price"), errors="coerce").fillna(0)
    df = df.dropna(subset=["Date"])

    prod = df[df["Product ID"] == product_id].groupby("Date", as_index=False).agg({
        "Units Sold": "sum", "Inventory Level": "sum", "Price": "mean",
        "Category": "first",
    }).sort_values("Date").reset_index(drop=True)

    prod["Lag_7"]           = prod["Units Sold"].shift(7)
    prod["Lag_30"]          = prod["Units Sold"].shift(30)
    prod["Rolling_Mean_7"]  = prod["Units Sold"].shift(1).rolling(7).mean()
    prod["Rolling_Mean_30"] = prod["Units Sold"].shift(1).rolling(30).mean()
    clean = prod.dropna().reset_index(drop=True)

    features = ["Lag_7", "Lag_30", "Rolling_Mean_7", "Rolling_Mean_30"]
    X = clean[features]
    y = clean["Units Sold"]
    split = int(len(X) * 0.85)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X[:split], y[:split])
    preds = model.predict(X)
    rmse  = float(np.sqrt(mean_squared_error(y[split:], model.predict(X[split:]))))
    acc   = float(max(0, 100 - (rmse / max(float(y.mean()), 1) * 100)))

    sigma        = float(prod["Units Sold"].std())
    safety_stock = int(round(1.65 * sigma * np.sqrt(7)))
    reorder_pt   = int(round(float(prod["Units Sold"].mean()) * 7 + safety_stock))
    avg_demand   = float(prod["Units Sold"].mean())

    # Build per-product report
    report = clean[["Date", "Units Sold", "Lag_7", "Lag_30", "Rolling_Mean_7", "Rolling_Mean_30"]].copy()
    report.insert(0, "Product_ID", product_id)
    report.insert(1, "Category",   str(clean["Category"].iloc[0]) if "Category" in clean.columns else "")
    report["Predicted_Demand"] = preds.round(0).astype(int)
    report["Safety_Stock"]     = safety_stock
    report["Reorder_Point"]    = reorder_pt
    report.columns = [
        "Product_ID", "Category", "Date", "Current_Demand",
        "Lag_7", "Lag_30", "Rolling_Mean_7", "Rolling_Mean_30",
        "Predicted_Demand", "Safety_Stock", "Reorder_Point",
    ]
    report["Date"] = report["Date"].dt.strftime("%Y-%m-%d")

    # Monthly demand range
    clean["Month"] = clean["Date"].dt.to_period("M")
    monthly = clean.groupby("Month")["Units Sold"].agg(
        Min_Demand="min", Max_Demand="max", Avg_Demand="mean"
    ).reset_index()
    monthly["Month"]      = monthly["Month"].astype(str)
    monthly["Avg_Demand"] = monthly["Avg_Demand"].round(0).astype(int)
    monthly.insert(0, "Product_ID", product_id)

    return report, monthly, clean, rmse, acc, safety_stock, reorder_pt, avg_demand

report, monthly, clean, rmse, acc, safety_stock, reorder_pt, avg_demand = forecast_product(
    selected_product, file_bytes
)


# ── KPIs ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### 📦 `{selected_product}` — {category}")
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Avg Daily Demand",  f"{int(avg_demand):,}")
with c2: st.metric("Safety Stock",      f"{safety_stock:,}")
with c3: st.metric("Reorder Point",     f"{reorder_pt:,}")
with c4: st.metric("Forecast Accuracy", f"{acc:.1f}%")
with c5: st.metric("RMSE",              f"{rmse:.1f}")
st.markdown("---")


# ── CHART 1: Actual vs Predicted ──────────────────────────────────────────────
st.subheader("📈 Actual vs Predicted Demand")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=pd.to_datetime(report["Date"]), y=report["Current_Demand"],
    name="Actual Demand", mode="lines",
    line=dict(color="#1d4ed8", width=2),
))
fig1.add_trace(go.Scatter(
    x=pd.to_datetime(report["Date"]), y=report["Predicted_Demand"],
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


# ── CHART 2: Monthly Demand Range ─────────────────────────────────────────────
st.subheader("📅 Monthly Demand Range — Min / Avg / Max")
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=monthly["Month"], y=monthly["Max_Demand"],
    name="Max Demand", marker_color="#bfdbfe",
))
fig2.add_trace(go.Bar(
    x=monthly["Month"], y=monthly["Avg_Demand"],
    name="Avg Demand", marker_color="#1d4ed8",
))
fig2.add_trace(go.Bar(
    x=monthly["Month"], y=monthly["Min_Demand"],
    name="Min Demand", marker_color="#93c5fd",
))
fig2.update_layout(
    barmode="overlay", template="plotly_white",
    xaxis=dict(showgrid=False, tickangle=-45),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=10, b=0), height=320,
)
st.plotly_chart(fig2, use_container_width=True)


# ── CHART 3: Rolling Means & Lags ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Rolling Mean (7-day & 30-day)")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=clean["Date"], y=clean["Rolling_Mean_7"],
        name="7-Day Rolling Mean", mode="lines",
        line=dict(color="#7c3aed", width=2),
    ))
    fig3.add_trace(go.Scatter(
        x=clean["Date"], y=clean["Rolling_Mean_30"],
        name="30-Day Rolling Mean", mode="lines",
        line=dict(color="#059669", width=2),
    ))
    fig3.update_layout(
        template="plotly_white",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=10, b=0), height=280,
    )
    st.plotly_chart(fig3, use_container_width=True)

with col2:
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=10, b=0), height=280,
    )
    st.plotly_chart(fig4, use_container_width=True)


# ── CHART 4: Inventory Level ──────────────────────────────────────────────────
st.subheader("🏭 Inventory Level Over Time")
fig5 = px.area(
    daily, x="Date", y="Inventory Level",
    color_discrete_sequence=["#0ea5e9"],
    template="plotly_white",
)
fig5.update_layout(
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    margin=dict(l=0, r=0, t=10, b=0), height=260,
)
st.plotly_chart(fig5, use_container_width=True)


# ── MONTHLY RANGE TABLE ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📅 Monthly Demand Summary Table")
st.markdown(f"Each row shows the **min / avg / max units sold per day** in that month for `{selected_product}`.")
st.dataframe(monthly, use_container_width=True, hide_index=True)


# ── RECOMMENDATION TABLE + DOWNLOAD ──────────────────────────────────────────
st.subheader("📋 Full Inventory Recommendation Table")
st.dataframe(report.tail(30), use_container_width=True, hide_index=True)

# Download: single product
csv_single = report.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"⬇️ Download Report for {selected_product}",
    data=csv_single,
    file_name=f"inventory_{selected_product}.csv",
    mime="text/csv",
    use_container_width=True,
)

st.markdown("---")

# Download: ALL products combined
st.subheader("📦 Download Report for All Products")
st.markdown("This generates forecasts for every product and combines them into one CSV.")

if st.button("Generate Full Report (All Products)", use_container_width=True):
    all_rows     = []
    all_monthly  = []
    progress     = st.progress(0, text="Processing products...")

    for i, pid in enumerate(all_products):
        try:
            r, m, _, _, _, _, _, _ = forecast_product(pid, file_bytes)
            all_rows.append(r)
            all_monthly.append(m)
        except Exception:
            pass
        progress.progress((i + 1) / len(all_products), text=f"Processing {pid}...")

    progress.empty()
    full_report   = pd.concat(all_rows,    ignore_index=True)
    full_monthly  = pd.concat(all_monthly, ignore_index=True)

    tab1, tab2 = st.tabs(["📋 Daily Recommendation Report", "📅 Monthly Range Summary"])
    with tab1:
        st.dataframe(full_report.head(50), use_container_width=True, hide_index=True)
        st.download_button(
            label="⬇️ Download Full Daily Report (All Products)",
            data=full_report.to_csv(index=False).encode("utf-8"),
            file_name="inventory_recommendations_all_products.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with tab2:
        st.dataframe(full_monthly, use_container_width=True, hide_index=True)
        st.download_button(
            label="⬇️ Download Monthly Range Report (All Products)",
            data=full_monthly.to_csv(index=False).encode("utf-8"),
            file_name="monthly_demand_range_all_products.csv",
            mime="text/csv",
            use_container_width=True,
        )
