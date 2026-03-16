import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
[data-testid="stSidebar"] .stRadio label { font-weight: 600; color: #c9d1d9; }
[data-testid="stMetricValue"] {
    font-size: 26px !important; font-weight: 700 !important;
    color: #58a6ff !important; font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricDelta"] { font-size: 13px !important; }
.card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;
}
.card-title {
    font-size: 12px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #8b949e; margin-bottom: 8px;
}
.alert-danger {
    background: rgba(248,81,73,0.12); border: 1px solid rgba(248,81,73,0.4);
    border-left: 4px solid #f85149; border-radius: 8px;
    padding: 14px 18px; color: #ffa198; margin-bottom: 10px; font-weight: 600;
}
.alert-warning {
    background: rgba(210,153,34,0.12); border: 1px solid rgba(210,153,34,0.4);
    border-left: 4px solid #d2991e; border-radius: 8px;
    padding: 14px 18px; color: #e3b341; margin-bottom: 10px; font-weight: 600;
}
.alert-success {
    background: rgba(63,185,80,0.12); border: 1px solid rgba(63,185,80,0.4);
    border-left: 4px solid #3fb950; border-radius: 8px;
    padding: 14px 18px; color: #56d364; margin-bottom: 10px; font-weight: 600;
}
.section-header {
    font-size: 22px; font-weight: 700; color: #e6edf3;
    margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid #21262d;
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
        df = pd.DataFrame({
            "Date": dates, "Category": categories,
            "Quantity": qty, "Price": prices * qty,
        })
        return df, None  # no raw_df for demo

    @staticmethod
    def load_csv(uploaded_file):
        """
        Returns:
          df     – daily aggregated df for charts/forecasting
          raw_df – full product-level rows for recommendation table
        CSV columns expected:
          Date, Store ID, Product ID, Category, Region, Inventory Level,
          Units Sold, Units Ordered, Price, Discount, Weather Condition,
          Promotion, Competitor Pricing, Seasonality, Epidemic, Demand
        """
        content = uploaded_file.read()

        # ── aggregated df ────────────────────────────────────────────────────
        df = pd.read_csv(io.BytesIO(content))
        df.columns = df.columns.str.strip().str.lstrip("\ufeff")
        rename = {}
        for c in df.columns:
            cl = c.lower().strip()
            if cl == "date":              rename[c] = "Date"
            elif cl == "units sold":      rename[c] = "Quantity"
            elif cl == "price":           rename[c] = "Price"
            elif cl == "category":        rename[c] = "Category"
            elif cl == "inventory level": rename[c] = "Inventory_Level"
        df = df.rename(columns=rename)

        if "Quantity" not in df.columns:
            st.error(
                f"Could not find 'Units Sold' column.\n\n"
                f"Columns found: **{', '.join(df.columns.tolist())}**"
            )
            st.stop()

        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"])
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0).astype(int)
        if "Price" in df.columns:
            df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)
        if "Inventory_Level" in df.columns:
            df["Inventory_Level"] = pd.to_numeric(df["Inventory_Level"], errors="coerce").fillna(0)

        agg = {"Quantity": "sum"}
        if "Price"           in df.columns: agg["Price"]           = "mean"
        if "Inventory_Level" in df.columns: agg["Inventory_Level"] = "sum"
        if "Category"        in df.columns: agg["Category"]        = "first"
        df = df.groupby("Date", as_index=False).agg(agg).sort_values("Date").reset_index(drop=True)

        # ── raw product-level df ─────────────────────────────────────────────
        raw = pd.read_csv(io.BytesIO(content))
        raw.columns = raw.columns.str.strip().str.lstrip("\ufeff")
        rn = {}
        for c in raw.columns:
            cl = c.lower().strip()
            if cl == "date":              rn[c] = "Date"
            elif cl == "product id":      rn[c] = "Product_ID"
            elif cl == "category":        rn[c] = "Category"
            elif cl == "units sold":      rn[c] = "Actual_Demand"
            elif cl == "inventory level": rn[c] = "Inventory_Level"
            elif cl == "price":           rn[c] = "Price"
        raw = raw.rename(columns=rn)
        raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
        raw = raw.dropna(subset=["Date"])
        if "Actual_Demand" in raw.columns:
            raw["Actual_Demand"] = pd.to_numeric(raw["Actual_Demand"], errors="coerce").fillna(0).astype(int)
        if "Inventory_Level" in raw.columns:
            raw["Inventory_Level"] = pd.to_numeric(raw["Inventory_Level"], errors="coerce").fillna(0)

        return df.sort_values("Date").reset_index(drop=True), raw.sort_values("Date").reset_index(drop=True)

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

        y_pred_test = model.predict(X[split:])
        mae  = mean_absolute_error(y[split:], y_pred_test) if split < len(X) else 0.0
        rmse = float(np.sqrt(mean_squared_error(y[split:], y_pred_test))) if split < len(X) else 0.0

        # In-sample predictions for recommendation table
        all_preds      = model.predict(X)
        in_sample_idx  = df.index.tolist()

        # Rolling 30-day future forecast
        history = list(series.values[-14:])
        future_preds = []
        for _ in range(horizon):
            lag1  = history[-1]
            lag7  = history[-7]  if len(history) >= 7  else history[0]
            rm7   = np.mean(history[-7:])
            rm14  = np.mean(history[-14:]) if len(history) >= 14 else np.mean(history)
            rstd7 = np.std(history[-7:])
            row   = pd.DataFrame([[lag1, lag7, rm7, rm14, rstd7]], columns=feature_cols)
            pred  = max(0, round(model.predict(row)[0]))
            future_preds.append(pred)
            history.append(pred)

        importances    = {k: float(v) for k, v in zip(feature_cols, model.feature_importances_)}
        all_preds_list = [float(v) for v in all_preds]
        in_sample_list = [int(i) for i in in_sample_idx]
        return future_preds, float(mae), float(rmse), importances, in_sample_list, all_preds_list

    @staticmethod
    def build_recommendation_table(raw_df, agg_df, in_sample_idx, all_preds,
                                   z_score=1.65, lead_time=7):
        """
        Build per-product-row recommendation table:
        Date, Product_ID, Category, Actual_Demand, Predicted_Demand,
        Safety_Stock, Reorder_Point, Suggested_Order
        """
        # Build a date -> predicted_demand map from in-sample predictions
        date_list = agg_df["Date"].tolist()
        pred_map  = {}
        for idx, pred in zip(in_sample_idx, all_preds):
            if idx < len(date_list):
                pred_map[date_list[idx]] = round(pred)

        # Fill any missing dates with nearest value
        all_dates = sorted(set(date_list))
        filled_pred = {}
        last = int(np.mean(list(pred_map.values()))) if pred_map else 50
        for d in all_dates:
            last = pred_map.get(d, last)
            filled_pred[d] = last

        result = raw_df.copy()
        result["Predicted_Demand"] = result["Date"].map(filled_pred).fillna(last).astype(int)

        # Safety stock and reorder point (global, based on full dataset)
        sigma         = result["Actual_Demand"].std() if "Actual_Demand" in result.columns else 10
        avg_demand    = result["Actual_Demand"].mean() if "Actual_Demand" in result.columns else 50
        safety_stock  = round(z_score * sigma * np.sqrt(lead_time))
        reorder_point = round(avg_demand * lead_time + safety_stock)

        result["Safety_Stock"]    = safety_stock
        result["Reorder_Point"]   = reorder_point
        result["Suggested_Order"] = reorder_point  # order up to reorder point

        # Final column selection and order
        keep = ["Date"]
        for col in ["Product_ID", "Category", "Actual_Demand", "Predicted_Demand",
                    "Safety_Stock", "Reorder_Point", "Suggested_Order"]:
            if col in result.columns:
                keep.append(col)

        return result[keep].reset_index(drop=True)

    @staticmethod
    def compute_alerts(df, forecast, low_threshold):
        alerts = []
        min_f = min(forecast)
        cur_q = int(df["Quantity"].iloc[-1])
        if min_f < low_threshold:
            alerts.append(("danger",
                f"🚨 STOCKOUT RISK — Forecasted demand drops to {min_f} units, "
                f"below your threshold of {low_threshold}."))
        if cur_q < low_threshold * 1.3:
            alerts.append(("warning",
                f"⚠️ LOW STOCK NOW — Current daily units sold ({cur_q}) is near the alert threshold."))
        if np.mean(forecast) > df["Quantity"].mean() * 1.2:
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
    df, raw_df = engine.load_csv(uploaded_file)
else:
    df, raw_df = engine.generate_demo_data()


# ── 6. FORECAST (cached) ──────────────────────────────────────────────────────
@st.cache_data
def run_forecast(qty_tuple):
    return InventoryEngine.train_and_forecast(pd.Series(list(qty_tuple)), horizon=30)

future_preds, mae, rmse, importances, in_sample_idx, all_preds = run_forecast(tuple(df["Quantity"].values))

last_date      = df["Date"].max()
forecast_dates = [last_date + timedelta(days=i + 1) for i in range(30)]
forecast_df    = pd.DataFrame({"Date": forecast_dates, "Forecast": future_preds})


# ── 7. DASHBOARD ──────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.markdown('<div class="section-header">Business Overview</div>', unsafe_allow_html=True)

    total_qty = int(df["Quantity"].sum())
    total_rev = df["Price"].sum()           if "Price"           in df.columns else 0
    total_inv = df["Inventory_Level"].sum() if "Inventory_Level" in df.columns else 0
    accuracy  = round(100 - (mae / max(df["Quantity"].mean(), 1) * 100), 1)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Units Sold",     f"{total_qty:,}",           "+8.3%")
    with c2: st.metric("Avg Daily Price",      f"₹{total_rev/max(len(df),1):,.2f}")
    with c3: st.metric("Total Inventory",      f"{int(total_inv):,}"       if total_inv else "N/A")
    with c4: st.metric("AI Forecast Accuracy", f"{accuracy:.1f}%",         f"RMSE: {rmse:.2f}")

    st.markdown("---")
    st.subheader("Daily Units Sold + 30-Day Forecast")
    fig = go.Figure([
        go.Scatter(
            x=df["Date"], y=df["Quantity"], name="Historical", mode="lines",
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
        upper = [p + mae for p in future_preds]
        lower = [max(0, p - mae) for p in future_preds]

        fig = go.Figure()
        hist_tail = df.tail(60)
        fig.add_trace(go.Scatter(
            x=hist_tail["Date"], y=hist_tail["Quantity"],
            name="Last 60 Days", mode="lines",
            line=dict(color="#58a6ff", width=2),
        ))
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

    with col2:
        avg_suggested = int(df["Quantity"].mean() * 7)
        st.markdown(f"""
        <div class="card">
            <div class="card-title">RMSE</div>
            <div style="font-size:28px;font-weight:700;color:#58a6ff;font-family:'JetBrains Mono',monospace">
                {rmse:.2f}
            </div>
        </div>
        <div class="card">
            <div class="card-title">Avg Predicted Demand</div>
            <div style="font-size:28px;font-weight:700;color:#3fb950;font-family:'JetBrains Mono',monospace">
                {int(np.mean(future_preds))}
            </div>
        </div>
        <div class="card">
            <div class="card-title">Avg Suggested Order</div>
            <div style="font-size:28px;font-weight:700;color:#e3b341;font-family:'JetBrains Mono',monospace">
                {avg_suggested}
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

    # ── Inventory Recommendation Table ────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Inventory Recommendation Table")

    if raw_df is not None:
        rec_table = engine.build_recommendation_table(
            raw_df, df, in_sample_idx, all_preds, z_score=1.65, lead_time=7
        )

        m1, m2, m3 = st.columns(3)
        with m1: st.metric("RMSE",                 f"{rmse:.2f}")
        with m2: st.metric("Avg Predicted Demand",  f"{int(np.mean(future_preds))}")
        with m3: st.metric("Avg Suggested Order",   f"{int(rec_table['Suggested_Order'].mean())}")

        st.dataframe(rec_table, use_container_width=True, hide_index=True)

        csv_bytes = rec_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download inventory_recommendations.csv",
            data=csv_bytes,
            file_name="inventory_recommendations.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("Upload a CSV to generate the recommendation table.")


# ── 9. ALERTS ─────────────────────────────────────────────────────────────────
elif page == "🚨 Alerts":
    st.markdown('<div class="section-header">Stock Alerts & Risk Monitor</div>', unsafe_allow_html=True)

    for level, msg in engine.compute_alerts(df, future_preds, low_threshold):
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
