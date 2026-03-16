import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import timedelta

# ── 1. PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OmniStream Inventory AI",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded",
)

# ── 2. GLOBAL CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stRadio label { font-weight: 600; color: #c9d1d9; }

[data-testid="stMetricValue"] {
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #58a6ff !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricDelta"] { font-size: 13px !important; }

.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.card-title {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 8px;
}
.alert-danger {
    background: rgba(248,81,73,0.12);
    border: 1px solid rgba(248,81,73,0.4);
    border-left: 4px solid #f85149;
    border-radius: 8px;
    padding: 14px 18px;
    color: #ffa198;
    margin-bottom: 10px;
    font-weight: 600;
}
.alert-warning {
    background: rgba(210,153,34,0.12);
    border: 1px solid rgba(210,153,34,0.4);
    border-left: 4px solid #d2991e;
    border-radius: 8px;
    padding: 14px 18px;
    color: #e3b341;
    margin-bottom: 10px;
    font-weight: 600;
}
.alert-success {
    background: rgba(63,185,80,0.12);
    border: 1px solid rgba(63,185,80,0.4);
    border-left: 4px solid #3fb950;
    border-radius: 8px;
    padding: 14px 18px;
    color: #56d364;
    margin-bottom: 10px;
    font-weight: 600;
}
.section-header {
    font-size: 22px;
    font-weight: 700;
    color: #e6edf3;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid #21262d;
}
</style>
""", unsafe_allow_html=True)


# ── 3. BACKEND ENGINE ─────────────────────────────────────────────────────────
class InventoryEngine:

    @staticmethod
    def generate_demo_data():
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", periods=365, freq="D")
        trend = np.linspace(50, 80, 365)
        seasonality = 15 * np.sin(np.linspace(0, 4 * np.pi, 365))
        noise = np.random.normal(0, 5, 365)
        qty = (trend + seasonality + noise).clip(5).astype(int)
        categories = np.random.choice(["Electronics", "Clothing", "Furniture", "Sports"], 365)
        prices = np.where(categories == "Electronics", 72.72,
                 np.where(categories == "Clothing", 80.16,
                 np.where(categories == "Furniture", 150.0, 60.0)))
        return pd.DataFrame({
            "Date": dates,
            "Category": categories,
            "Quantity": qty,
            "Price": prices * qty,
        })

    @staticmethod
    def load_csv(uploaded_file):
        df = pd.read_csv(uploaded_file)
        # Strip BOM and whitespace from column names
        df.columns = df.columns.str.strip().str.lstrip("\ufeff")

        # ── Map your actual CSV columns ──────────────────────────────────────
        # CSV has: Date, Store ID, Product ID, Category, Region,
        #          Inventory Level, Units Sold, Units Ordered, Price,
        #          Discount, Weather Condition, Promotion,
        #          Competitor Pricing, Seasonality, Epidemic, Demand
        rename = {}
        for c in df.columns:
            cl = c.lower().strip()
            if cl == "date":
                rename[c] = "Date"
            elif cl == "units sold":
                rename[c] = "Quantity"
            elif cl == "price":
                rename[c] = "Price"
            elif cl == "category":
                rename[c] = "Category"
            elif cl == "inventory level":
                rename[c] = "Inventory Level"
            elif cl == "demand":
                rename[c] = "Demand"

        df = df.rename(columns=rename)

        # Validate
        if "Quantity" not in df.columns:
            st.error(
                f"Could not map a Quantity column.\n\n"
                f"Columns in your CSV: **{', '.join(df.columns.tolist())}**\n\n"
                "Expecting a column named `Units Sold`."
            )
            st.stop()

        # Parse dates — DD-MM-YYYY format
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"])

        # Coerce numerics
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0).astype(int)
        if "Price" in df.columns:
            df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)
        if "Inventory Level" in df.columns:
            df["Inventory Level"] = pd.to_numeric(df["Inventory Level"], errors="coerce").fillna(0)

        # Aggregate by date (sum across stores/products)
        agg = {"Quantity": "sum"}
        if "Price" in df.columns:        agg["Price"] = "mean"
        if "Inventory Level" in df.columns: agg["Inventory Level"] = "sum"
        if "Category" in df.columns:     agg["Category"] = "first"

        df = df.groupby("Date", as_index=False).agg(agg)
        return df.sort_values("Date").reset_index(drop=True)

    @staticmethod
    def train_and_forecast(series: pd.Series, horizon: int = 30):
        df = series.reset_index(drop=True).to_frame(name="Quantity")
        df["Lag_1"]           = df["Quantity"].shift(1)
        df["Lag_7"]           = df["Quantity"].shift(7)
        df["Rolling_Mean_7"]  = df["Quantity"].shift(1).rolling(7).mean()
        df["Rolling_Mean_14"] = df["Quantity"].shift(1).rolling(14).mean()
        df["Rolling_Std_7"]   = df["Quantity"].shift(1).rolling(7).std()
        df = df.dropna()

        feature_cols = ["Lag_1", "Lag_7", "Rolling_Mean_7", "Rolling_Mean_14", "Rolling_Std_7"]
        X, y = df[feature_cols], df["Quantity"]

        split = int(len(X) * 0.85)
        model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        model.fit(X[:split], y[:split])
        mae = mean_absolute_error(y[split:], model.predict(X[split:])) if split < len(X) else 0.0

        history = list(series.values[-14:])
        preds = []
        for _ in range(horizon):
            lag1  = history[-1]
            lag7  = history[-7]  if len(history) >= 7  else history[0]
            rm7   = np.mean(history[-7:])
            rm14  = np.mean(history[-14:]) if len(history) >= 14 else np.mean(history)
            rstd7 = np.std(history[-7:])
            row   = pd.DataFrame([[lag1, lag7, rm7, rm14, rstd7]], columns=feature_cols)
            pred  = max(0, round(model.predict(row)[0]))
            preds.append(pred)
            history.append(pred)

        importances = dict(zip(feature_cols, model.feature_importances_))
        return preds, mae, importances

    @staticmethod
    def compute_alerts(df, forecast, low_threshold):
        alerts = []
        min_f = min(forecast)
        avg_f = np.mean(forecast)
        cur_q = df["Quantity"].iloc[-1]

        if min_f < low_threshold:
            alerts.append(("danger",
                f"🚨 STOCKOUT RISK — Forecasted demand drops to {min_f} units, "
                f"below your threshold of {low_threshold}."))
        if cur_q < low_threshold * 1.3:
            alerts.append(("warning",
                f"⚠️ LOW STOCK NOW — Current daily units sold ({cur_q}) is near the alert threshold."))
        if avg_f > df["Quantity"].mean() * 1.2:
            alerts.append(("warning",
                "📈 DEMAND SURGE — Forecasted demand is 20%+ above historical average. Consider restocking early."))
        if not alerts:
            alerts.append(("success", "✅ All Clear — No stockout risk detected in the 30-day forecast."))
        return alerts


# ── 4. SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ OmniStream AI")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "🤖 Demand Forecasting", "🚨 Alerts", "📐 Inventory Logic"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type="csv")
    st.markdown("---")
    low_threshold = st.slider("Low Stock Threshold (units/day)", 5, 500, 50)
    if not uploaded_file:
        st.info("No CSV uploaded — using demo data.")


# ── 5. LOAD DATA ──────────────────────────────────────────────────────────────
engine = InventoryEngine()

if uploaded_file:
    df = engine.load_csv(uploaded_file)
else:
    df = engine.generate_demo_data()


# ── 6. FORECAST (cached) ──────────────────────────────────────────────────────
@st.cache_data
def run_forecast(qty_tuple):
    return InventoryEngine.train_and_forecast(pd.Series(list(qty_tuple)), horizon=30)

forecast_preds, mae, importances = run_forecast(tuple(df["Quantity"].values))

last_date      = df["Date"].max()
forecast_dates = [last_date + timedelta(days=i + 1) for i in range(30)]
forecast_df    = pd.DataFrame({"Date": forecast_dates, "Forecast": forecast_preds})


# ── 7. DASHBOARD ──────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.markdown('<div class="section-header">Business Overview</div>', unsafe_allow_html=True)

    total_qty = int(df["Quantity"].sum())
    total_rev = df["Price"].sum()             if "Price"           in df.columns else 0
    total_inv = df["Inventory Level"].sum()   if "Inventory Level" in df.columns else 0
    accuracy  = round(100 - (mae / max(df["Quantity"].mean(), 1) * 100), 1)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Units Sold",      f"{total_qty:,}",        "+8.3%")
    with c2: st.metric("Avg Price",             f"₹{total_rev/max(len(df),1):,.2f}")
    with c3: st.metric("Total Inventory",       f"{int(total_inv):,}"    if total_inv else "N/A")
    with c4: st.metric("AI Forecast Accuracy",  f"{accuracy:.1f}%",      f"MAE: {mae:.1f} units")

    st.markdown("---")

    st.subheader("Daily Units Sold + 30-Day Forecast")
    fig = go.Figure([
        go.Scatter(
            x=df["Date"], y=df["Quantity"],
            name="Historical", mode="lines",
            line=dict(color="#58a6ff", width=2),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.08)"
        ),
        go.Scatter(
            x=forecast_df["Date"], y=forecast_df["Forecast"],
            name="30-Day Forecast", mode="lines",
            line=dict(color="#3fb950", width=2, dash="dot"),
            fill="tozeroy", fillcolor="rgba(63,185,80,0.06)"
        ),
    ])
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262d"),
        legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=0, r=0, t=10, b=0), height=340,
    )
    st.plotly_chart(fig, use_container_width=True)

    if "Category" in df.columns:
        st.subheader("Units Sold by Category")
        cat_df = (df.groupby("Category")["Quantity"].sum()
                    .reset_index().sort_values("Quantity", ascending=True))
        fig2 = px.bar(cat_df, x="Quantity", y="Category", orientation="h",
                      color="Quantity", color_continuous_scale="Blues", template="plotly_dark")
        fig2.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                           coloraxis_showscale=False, margin=dict(l=0,r=0,t=10,b=0), height=280)
        st.plotly_chart(fig2, use_container_width=True)


# ── 8. DEMAND FORECASTING ─────────────────────────────────────────────────────
elif page == "🤖 Demand Forecasting":
    st.markdown('<div class="section-header">AI Forecasting Engine</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("30-Day Rolling Demand Prediction")
        upper = [p + mae for p in forecast_preds]
        lower = [max(0, p - mae) for p in forecast_preds]

        fig = go.Figure()
        hist_tail = df.tail(60)
        fig.add_trace(go.Scatter(
            x=hist_tail["Date"], y=hist_tail["Quantity"],
            name="Last 60 Days", mode="lines",
            line=dict(color="#58a6ff", width=2),
        ))
        # Confidence band
        fig.add_trace(go.Scatter(
            x=list(forecast_df["Date"]) + list(forecast_df["Date"])[::-1],
            y=upper + lower[::-1],
            fill="toself", fillcolor="rgba(63,185,80,0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="Confidence Band",
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["Date"], y=forecast_df["Forecast"],
            name="Forecast", mode="lines+markers",
            line=dict(color="#3fb950", width=2.5), marker=dict(size=5),
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262d"),
            legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=0,r=0,t=10,b=0), height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Full 30-Day Forecast Table"):
            tbl = forecast_df.copy()
            tbl["Date"]        = tbl["Date"].dt.strftime("%b %d, %Y")
            tbl["Forecast"]    = tbl["Forecast"].astype(int)
            tbl["Lower Bound"] = [max(0, p - int(mae)) for p in forecast_preds]
            tbl["Upper Bound"] = [p + int(mae)         for p in forecast_preds]
            st.dataframe(tbl, use_container_width=True, hide_index=True)

    with col2:
        avg_acc = round(100 - (mae / max(df["Quantity"].mean(), 1) * 100), 1)
        peak_date = forecast_df.loc[forecast_df["Forecast"].idxmax(), "Date"].strftime("%b %d")
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Mean Absolute Error</div>
            <div style="font-size:28px;font-weight:700;color:#58a6ff;font-family:'JetBrains Mono',monospace">
                {mae:.1f} units
            </div>
        </div>
        <div class="card">
            <div class="card-title">Forecast Accuracy</div>
            <div style="font-size:28px;font-weight:700;color:#3fb950;font-family:'JetBrains Mono',monospace">
                {avg_acc}%
            </div>
        </div>
        <div class="card">
            <div class="card-title">Peak Forecast Day</div>
            <div style="font-size:22px;font-weight:700;color:#e3b341;font-family:'JetBrains Mono',monospace">
                {peak_date}
                <span style="font-size:14px;color:#8b949e">({max(forecast_preds)} units)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Feature Importance")
        feat_df = (pd.DataFrame(list(importances.items()), columns=["Feature", "Importance"])
                     .sort_values("Importance"))
        fig3 = px.bar(feat_df, x="Importance", y="Feature", orientation="h",
                      color="Importance", color_continuous_scale="Blues", template="plotly_dark")
        fig3.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                           coloraxis_showscale=False, margin=dict(l=0,r=0,t=10,b=0), height=240)
        st.plotly_chart(fig3, use_container_width=True)


# ── 9. ALERTS ─────────────────────────────────────────────────────────────────
elif page == "🚨 Alerts":
    st.markdown('<div class="section-header">Stock Alerts & Risk Monitor</div>', unsafe_allow_html=True)

    for level, msg in engine.compute_alerts(df, forecast_preds, low_threshold):
        st.markdown(f'<div class="alert-{level}">{msg}</div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Forecast vs. Threshold")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=forecast_df["Date"], y=forecast_df["Forecast"],
            name="Forecast", mode="lines+markers",
            line=dict(color="#58a6ff", width=2), marker=dict(size=5),
        ))
        fig4.add_hline(y=low_threshold, line_dash="dot", line_color="#f85149",
                       annotation_text=f"Threshold: {low_threshold}",
                       annotation_position="top left")
        fig4.update_layout(
            template="plotly_dark", paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262d"),
            margin=dict(l=0,r=0,t=10,b=0), height=300,
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        st.subheader("30-Day Demand Distribution")
        fig5 = px.histogram(forecast_df, x="Forecast", nbins=15,
                            color_discrete_sequence=["#58a6ff"], template="plotly_dark")
        fig5.add_vline(x=low_threshold, line_dash="dot", line_color="#f85149",
                       annotation_text="Threshold")
        fig5.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                           margin=dict(l=0,r=0,t=10,b=0), height=300)
        st.plotly_chart(fig5, use_container_width=True)

    at_risk = forecast_df[forecast_df["Forecast"] < low_threshold]
    if not at_risk.empty:
        st.subheader(f"🚨 {len(at_risk)} At-Risk Days")
        ar = at_risk.copy()
        ar["Date"]     = ar["Date"].dt.strftime("%b %d, %Y")
        ar["Forecast"] = ar["Forecast"].astype(int)
        ar["Gap"]      = low_threshold - ar["Forecast"]
        st.dataframe(ar, use_container_width=True, hide_index=True)
    else:
        st.success("No days in the 30-day forecast fall below the threshold.")


# ── 10. INVENTORY LOGIC ───────────────────────────────────────────────────────
elif page == "📐 Inventory Logic":
    st.markdown('<div class="section-header">Inventory Optimization Logic</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Safety Stock Calculator")
        st.latex(r"SS = Z \times \sigma_{LT} \times \sqrt{L}")

        z_score   = st.slider("Service Level Z-Score", 1.0, 3.0, 1.65, 0.05,
                              help="1.65 = 95%,  2.05 = 98%,  2.33 = 99%")
        sigma_lt  = st.slider("Demand Std Dev during Lead Time (σ)", 1, 500,
                              int(df["Quantity"].std()) or 20)
        lead_time = st.slider("Lead Time (days)", 1, 30, 7)

        ss  = round(z_score * sigma_lt * np.sqrt(lead_time))
        rop = round(df["Quantity"].mean() * lead_time + ss)

        st.markdown(f"""
        <div class="card">
            <div class="card-title">Safety Stock</div>
            <div style="font-size:32px;font-weight:700;color:#3fb950;font-family:'JetBrains Mono',monospace">
                {ss} units
            </div>
        </div>
        <div class="card">
            <div class="card-title">Reorder Point</div>
            <div style="font-size:32px;font-weight:700;color:#58a6ff;font-family:'JetBrains Mono',monospace">
                {rop} units
            </div>
            <div style="font-size:12px;color:#8b949e;margin-top:6px">
                Avg daily demand × lead time + safety stock
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.subheader("EOQ Calculator")
        st.latex(r"EOQ = \sqrt{\frac{2DS}{H}}")

        annual_demand = st.number_input("Annual Demand (D, units)",
                                        value=int(df["Quantity"].sum()), step=500)
        order_cost    = st.number_input("Ordering Cost (S, ₹ per order)", value=500, step=50)
        holding_cost  = st.number_input("Holding Cost (H, ₹ per unit/year)", value=20, step=5)

        eoq           = round(np.sqrt((2 * annual_demand * order_cost) / max(holding_cost, 1)))
        orders_per_yr = round(annual_demand / max(eoq, 1), 1)
        cycle_days    = round(365 / max(orders_per_yr, 0.01))

        st.markdown(f"""
        <div class="card">
            <div class="card-title">Optimal Order Quantity (EOQ)</div>
            <div style="font-size:32px;font-weight:700;color:#e3b341;font-family:'JetBrains Mono',monospace">
                {eoq} units
            </div>
        </div>
        <div class="card">
            <div class="card-title">Orders per Year / Cycle Days</div>
            <div style="font-size:28px;font-weight:700;color:#58a6ff;font-family:'JetBrains Mono',monospace">
                {orders_per_yr}x
                <span style="font-size:16px;color:#8b949e">/ {cycle_days} days</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
