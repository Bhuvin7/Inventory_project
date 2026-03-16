import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import timedelta

# ── 1. PAGE CONFIG ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OmniStream Inventory AI",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded",
)

# ── 2. GLOBAL CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* Base */
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stRadio label { font-weight: 600; color: #c9d1d9; }

/* Metric tiles */
[data-testid="stMetricValue"] {
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #58a6ff !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricDelta"] { font-size: 13px !important; }

/* Cards */
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

/* Alert banners */
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

/* Section headers */
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


# ── 3. BACKEND ENGINE ────────────────────────────────────────────────────────
class InventoryEngine:

    @staticmethod
    def generate_demo_data():
        """Generate realistic demo data if no CSV uploaded."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=365, freq="D")
        trend = np.linspace(50, 80, 365)
        seasonality = 15 * np.sin(np.linspace(0, 4 * np.pi, 365))
        noise = np.random.normal(0, 5, 365)
        qty = (trend + seasonality + noise).clip(5).astype(int)

        products = np.random.choice(["Widget A", "Widget B", "Gadget X", "Gadget Y"], 365)
        prices   = np.where(products == "Widget A", 29.99,
                   np.where(products == "Widget B", 49.99,
                   np.where(products == "Gadget X", 89.99, 119.99)))

        return pd.DataFrame({
            "Date": dates,
            "Product": products,
            "Quantity": qty,
            "Price": prices * qty,
        })

    @staticmethod
    def train_and_forecast(series: pd.Series, horizon: int = 30):
        """
        Train a RandomForest on lag/rolling features and
        roll forward `horizon` days of predictions.
        Returns (forecast_series, mae, feature_importances_dict).
        """
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
        mae = mean_absolute_error(y[split:], model.predict(X[split:]))

        # Rolling forecast
        history = list(series.values[-14:])
        preds = []
        for _ in range(horizon):
            lag1  = history[-1]
            lag7  = history[-7] if len(history) >= 7  else history[0]
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
    def compute_alerts(df: pd.DataFrame, forecast: list, low_threshold: int):
        alerts = []
        min_forecast = min(forecast)
        avg_forecast = np.mean(forecast)
        current_qty  = df["Quantity"].iloc[-1]

        if min_forecast < low_threshold:
            alerts.append(("danger",
                f"🚨 STOCKOUT RISK — Forecasted demand dips to {min_forecast} units, "
                f"below your threshold of {low_threshold}."))
        if current_qty < low_threshold * 1.3:
            alerts.append(("warning",
                f"⚠️ LOW STOCK NOW — Current quantity ({current_qty}) is near the alert threshold."))
        if avg_forecast > df["Quantity"].mean() * 1.2:
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
    low_threshold = st.slider("Low Stock Threshold (units)", 5, 100, 20)

    if not uploaded_file:
        st.info("No CSV? Using demo data.")


# ── 5. LOAD DATA ──────────────────────────────────────────────────────────────
engine = InventoryEngine()

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Flexible column detection
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    qty_col  = next((c for c in df.columns if "qty" in c.lower() or "quantity" in c.lower()), None)
    rev_col  = next((c for c in df.columns if "price" in c.lower() or "revenue" in c.lower() or "sales" in c.lower()), None)
    prod_col = next((c for c in df.columns if "product" in c.lower()), None)

    rename = {}
    if date_col: rename[date_col] = "Date"
    if qty_col:  rename[qty_col]  = "Quantity"
    if rev_col:  rename[rev_col]  = "Price"
    if prod_col: rename[prod_col] = "Product"
    df = df.rename(columns=rename)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
else:
    df = engine.generate_demo_data()
    st.sidebar.caption("📡 Demo mode active")


# ── 6. RUN FORECAST (cached per data) ─────────────────────────────────────────
@st.cache_data
def run_forecast(qty_tuple):
    series = pd.Series(list(qty_tuple))
    return engine.train_and_forecast(series, horizon=30)

forecast_preds, mae, importances = run_forecast(tuple(df["Quantity"].values))

last_date      = df["Date"].max()
forecast_dates = [last_date + timedelta(days=i+1) for i in range(30)]
forecast_df    = pd.DataFrame({"Date": forecast_dates, "Forecast": forecast_preds})


# ── 7. PAGE: DASHBOARD ────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.markdown('<div class="section-header">Business Overview</div>', unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    total_qty = df["Quantity"].sum()
    total_rev = df["Price"].sum() if "Price" in df.columns else 0
    turnover  = round(df["Quantity"].sum() / max(df["Quantity"].mean(), 1), 1)
    accuracy  = round(100 - (mae / df["Quantity"].mean() * 100), 1)

    with c1: st.metric("Total Units Sold",     f"{total_qty:,}",              "+8.3%")
    with c2: st.metric("Total Revenue",        f"₹{total_rev:,.0f}",          "+11.2%")
    with c3: st.metric("Inventory Turnover",   f"{turnover}x")
    with c4: st.metric("AI Forecast Accuracy", f"{accuracy:.1f}%",            delta="Model MAE: " + str(round(mae, 1)))

    st.markdown("---")

    # Historical + Forecast Overlay
    st.subheader("Inventory Velocity + 30-Day Forecast")
    hist_trace = go.Scatter(
        x=df["Date"], y=df["Quantity"],
        name="Historical", mode="lines",
        line=dict(color="#58a6ff", width=2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)"
    )
    fore_trace = go.Scatter(
        x=forecast_df["Date"], y=forecast_df["Forecast"],
        name="30-Day Forecast", mode="lines",
        line=dict(color="#3fb950", width=2, dash="dot"),
        fill="tozeroy", fillcolor="rgba(63,185,80,0.06)"
    )
    fig = go.Figure([hist_trace, fore_trace])
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262d"),
        legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=0, r=0, t=10, b=0),
        height=340,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Product breakdown (if available)
    if "Product" in df.columns:
        st.subheader("Sales by Product")
        prod_df = df.groupby("Product")["Quantity"].sum().reset_index().sort_values("Quantity", ascending=True)
        fig2 = px.bar(prod_df, x="Quantity", y="Product", orientation="h",
                      color="Quantity", color_continuous_scale="Blues",
                      template="plotly_dark")
        fig2.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                           coloraxis_showscale=False, margin=dict(l=0,r=0,t=10,b=0), height=260)
        st.plotly_chart(fig2, use_container_width=True)


# ── 8. PAGE: DEMAND FORECASTING ───────────────────────────────────────────────
elif page == "🤖 Demand Forecasting":
    st.markdown('<div class="section-header">AI Forecasting Engine</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("30-Day Rolling Demand Prediction")

        fig = go.Figure()
        # Last 60 days historical
        hist_tail = df.tail(60)
        fig.add_trace(go.Scatter(
            x=hist_tail["Date"], y=hist_tail["Quantity"],
            name="Last 60 Days", mode="lines",
            line=dict(color="#58a6ff", width=2),
        ))
        # Forecast band
        upper = [p + mae for p in forecast_preds]
        lower = [max(0, p - mae) for p in forecast_preds]
        fig.add_trace(go.Scatter(
            x=forecast_df["Date"] + forecast_df["Date"][::-1].values.tolist(),
            y=upper + lower[::-1],
            fill="toself", fillcolor="rgba(63,185,80,0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="Confidence Band", showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["Date"], y=forecast_df["Forecast"],
            name="Forecast", mode="lines+markers",
            line=dict(color="#3fb950", width=2.5),
            marker=dict(size=5),
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262d"),
            legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=0,r=0,t=10,b=0), height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        with st.expander("📋 Full 30-Day Forecast Table"):
            display_df = forecast_df.copy()
            display_df["Date"]        = display_df["Date"].dt.strftime("%b %d, %Y")
            display_df["Forecast"]    = display_df["Forecast"].astype(int)
            display_df["Lower Bound"] = [max(0, p - int(mae)) for p in forecast_preds]
            display_df["Upper Bound"] = [p + int(mae) for p in forecast_preds]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Model Metrics")
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
                {round(100 - (mae / df["Quantity"].mean() * 100), 1)}%
            </div>
        </div>
        <div class="card">
            <div class="card-title">Peak Forecast Day</div>
            <div style="font-size:22px;font-weight:700;color:#e3b341;font-family:'JetBrains Mono',monospace">
                {forecast_df.loc[forecast_df['Forecast'].idxmax(), 'Date'].strftime('%b %d')}
                &nbsp;<span style="font-size:14px;color:#8b949e">({max(forecast_preds)} units)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Feature Importance")
        feat_df = pd.DataFrame(list(importances.items()), columns=["Feature", "Importance"]).sort_values("Importance")
        fig3 = px.bar(feat_df, x="Importance", y="Feature", orientation="h",
                      color="Importance", color_continuous_scale="Blues",
                      template="plotly_dark")
        fig3.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                           coloraxis_showscale=False, margin=dict(l=0,r=0,t=10,b=0), height=240)
        st.plotly_chart(fig3, use_container_width=True)


# ── 9. PAGE: ALERTS ───────────────────────────────────────────────────────────
elif page == "🚨 Alerts":
    st.markdown('<div class="section-header">Stock Alerts & Risk Monitor</div>', unsafe_allow_html=True)

    alerts = engine.compute_alerts(df, forecast_preds, low_threshold)

    for level, msg in alerts:
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
                       annotation_text=f"Threshold: {low_threshold}", annotation_position="top left")
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

    # Days at risk
    at_risk = forecast_df[forecast_df["Forecast"] < low_threshold]
    if not at_risk.empty:
        st.subheader(f"🚨 {len(at_risk)} At-Risk Days")
        at_risk_display = at_risk.copy()
        at_risk_display["Date"]     = at_risk_display["Date"].dt.strftime("%b %d, %Y")
        at_risk_display["Forecast"] = at_risk_display["Forecast"].astype(int)
        at_risk_display["Gap"]      = low_threshold - at_risk_display["Forecast"]
        st.dataframe(at_risk_display, use_container_width=True, hide_index=True)
    else:
        st.success("No days in the 30-day forecast fall below the threshold.")


# ── 10. PAGE: INVENTORY LOGIC ─────────────────────────────────────────────────
elif page == "📐 Inventory Logic":
    st.markdown('<div class="section-header">Inventory Optimization Logic</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Safety Stock Calculator")
        st.markdown("Uses the formula:")
        st.latex(r"SS = Z \times \sigma_{LT} \times \sqrt{L}")

        z_score     = st.slider("Service Level Z-Score", 1.0, 3.0, 1.65, 0.05,
                                help="1.65 = 95%, 2.05 = 98%, 2.33 = 99%")
        sigma_lt    = st.slider("Demand Std Dev during Lead Time (σ)", 1, 50, int(df["Quantity"].std()))
        lead_time   = st.slider("Lead Time (days)", 1, 30, 7)

        safety_stock = round(z_score * sigma_lt * np.sqrt(lead_time))
        avg_demand   = round(df["Quantity"].mean())
        reorder_pt   = round(avg_demand * lead_time + safety_stock)

        st.markdown(f"""
        <div class="card">
            <div class="card-title">Safety Stock</div>
            <div style="font-size:32px;font-weight:700;color:#3fb950;font-family:'JetBrains Mono',monospace">
                {safety_stock} units
            </div>
        </div>
        <div class="card">
            <div class="card-title">Reorder Point</div>
            <div style="font-size:32px;font-weight:700;color:#58a6ff;font-family:'JetBrains Mono',monospace">
                {reorder_pt} units
            </div>
            <div style="font-size:12px;color:#8b949e;margin-top:6px">
                Avg daily demand × lead time + safety stock
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.subheader("EOQ Calculator")
        st.markdown("Economic Order Quantity:")
        st.latex(r"EOQ = \sqrt{\frac{2DS}{H}}")

        annual_demand  = st.number_input("Annual Demand (D, units)", value=int(df["Quantity"].sum()), step=100)
        order_cost     = st.number_input("Ordering Cost (S, ₹ per order)", value=500, step=50)
        holding_cost   = st.number_input("Holding Cost (H, ₹ per unit/year)", value=20, step=5)

        eoq = round(np.sqrt((2 * annual_demand * order_cost) / max(holding_cost, 1)))
        orders_per_yr = round(annual_demand / max(eoq, 1), 1)
        cycle_days    = round(365 / max(orders_per_yr, 0.01))

        st.markdown(f"""
        <div class="card">
            <div class="card-title">Optimal Order Quantity</div>
            <div style="font-size:32px;font-weight:700;color:#e3b341;font-family:'JetBrains Mono',monospace">
                {eoq} units
            </div>
        </div>
        <div class="card">
            <div class="card-title">Orders per Year / Cycle Days</div>
            <div style="font-size:28px;font-weight:700;color:#58a6ff;font-family:'JetBrains Mono',monospace">
                {orders_per_yr}x &nbsp;<span style="font-size:16px;color:#8b949e">/ {cycle_days} days</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
