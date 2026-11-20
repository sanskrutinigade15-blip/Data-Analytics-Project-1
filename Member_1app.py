"""
Causal Impact Visualization
File : app.py
Data : causal_impact_data.csv (same folder as this file)

Run from that folder in the Terminal:
    python -m streamlit run app.py --server.port 8051
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="Causal Impact Visualization", layout="wide")
st.title("Causal Impact Visualization")
st.write(
    "Estimates what would have happened **without** an intervention, using "
    "control series from the period before it, and compares that to what "
    "actually happened."
)

CSV_PATH = Path(__file__).parent / "causal_impact_data.csv"
NO_DATE = "(use row number)"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def make_sample_data(n=120, intervention=90, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    x1 = 100 + np.cumsum(rng.normal(0, 1, n))
    x2 = 50 + np.cumsum(rng.normal(0, 0.7, n))
    y = 0.6 * x1 + 0.8 * x2 + rng.normal(0, 1, n)
    y[intervention:] += 10  # true lift after the intervention
    return pd.DataFrame({"date": dates, "y": y, "x1": x1, "x2": x2})


def load_data(uploaded):
    if uploaded is not None:
        return pd.read_csv(uploaded), f"Loaded uploaded file: {uploaded.name}"
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH), f"Loaded {CSV_PATH.name}"
    return make_sample_data(), "CSV not found - showing sample data"


def run_causal_impact(y, X, split):
    """Fit OLS on the pre-period, predict the counterfactual for all rows."""
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A[:split], y[:split], rcond=None)
    pred = A @ beta
    sigma = np.std(y[:split] - pred[:split], ddof=A.shape[1])
    lower, upper = pred - 1.96 * sigma, pred + 1.96 * sigma
    return pred, lower, upper


# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
df, msg = load_data(uploaded_file)
st.caption(msg)

with st.expander("Preview data"):
    st.dataframe(df.head(20), use_container_width=True)

# ------------------------------------------------------------------
# Sidebar settings
# ------------------------------------------------------------------
st.sidebar.header("Settings")

date_options = [NO_DATE] + list(df.columns)
default_date = "date" if "date" in df.columns else NO_DATE
date_col = st.sidebar.selectbox(
    "Date column", date_options, index=date_options.index(default_date)
)

numeric_cols = [c for c in df.select_dtypes(include=np.number).columns]
if len(numeric_cols) < 2:
    st.error("Need at least two numeric columns (one outcome + one control).")
    st.stop()

y_default = "y" if "y" in numeric_cols else numeric_cols[0]
y_col = st.sidebar.selectbox(
    "Outcome column", numeric_cols, index=numeric_cols.index(y_default)
)

control_options = [c for c in numeric_cols if c != y_col]
controls = st.sidebar.multiselect(
    "Control columns", control_options, default=control_options
)
if not controls:
    st.warning("Pick at least one control column.")
    st.stop()

n = len(df)
split = st.sidebar.slider(
    "Intervention starts at row", min_value=10, max_value=n - 5,
    value=min(int(n * 0.75), n - 5),
)

# x-axis values
if date_col == NO_DATE:
    x_axis = np.arange(n)
else:
    x_axis = pd.to_datetime(df[date_col], errors="coerce")
    if x_axis.isna().all():
        x_axis = np.arange(n)

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
data = df[[y_col] + controls].astype(float).interpolate().bfill().ffill()
y = data[y_col].to_numpy()
X = data[controls].to_numpy()

pred, lower, upper = run_causal_impact(y, X, split)
effect = y - pred
cum_effect = np.cumsum(np.where(np.arange(n) >= split, effect, 0.0))

post = slice(split, n)
avg_actual = y[post].mean()
avg_pred = pred[post].mean()
abs_effect = avg_actual - avg_pred
rel_effect = abs_effect / avg_pred * 100
total_effect = effect[post].sum()

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg actual (post)", f"{avg_actual:.2f}")
c2.metric("Avg predicted (post)", f"{avg_pred:.2f}")
c3.metric("Absolute effect", f"{abs_effect:.2f}")
c4.metric("Relative effect", f"{rel_effect:.1f}%")
st.write(f"**Cumulative effect over the post-period:** {total_effect:.2f}")

# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
cut = x_axis[split] if not isinstance(x_axis, np.ndarray) else split

axes[0].plot(x_axis, y, label="Actual", color="black")
axes[0].plot(x_axis, pred, "--", label="Predicted (no intervention)", color="tab:blue")
axes[0].fill_between(x_axis, lower, upper, color="tab:blue", alpha=0.2, label="95% interval")
axes[0].set_title("Actual vs predicted")

axes[1].plot(x_axis, effect, color="tab:green")
axes[1].axhline(0, color="gray", lw=1)
axes[1].fill_between(x_axis, lower - pred, upper - pred, color="tab:blue", alpha=0.2)
axes[1].set_title("Pointwise effect")

axes[2].plot(x_axis, cum_effect, color="tab:red")
axes[2].axhline(0, color="gray", lw=1)
axes[2].set_title("Cumulative effect")

for ax in axes:
    ax.axvline(cut, color="red", linestyle=":", label="Intervention")
    ax.grid(alpha=0.3)
axes[0].legend(loc="upper left")
fig.tight_layout()
st.pyplot(fig)

st.divider()

st.header("Pricing-Specific Revenue Analysis")

st.write(
    "Compare estimated baseline revenue "
    "with counterfactual revenue under "
    "different pricing scenarios."
)
# Pricing scenario controls
st.subheader("Adjust Pricing Scenario")

price_change = st.slider(
    "Price Change (%)",
    min_value=-30,
    max_value=30,
    value=10,
    step=5
)

elasticity = st.slider(
    "Demand Price Elasticity",
    min_value=0.1,
    max_value=3.0,
    value=1.5,
    step=0.1
)

baseline_demand = st.slider(
    "Assumed Baseline Sales per Property",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1
)

# Prepare pricing data
price_data = pd.to_numeric(
    df["Price (INR)"], errors="coerce"
).dropna()

# Calculate baseline and counterfactual values
baseline_price = price_data.mean()

new_price = baseline_price * (1 + price_change / 100)

demand_factor = (
    new_price / baseline_price
) ** (-elasticity)

estimated_baseline_units = len(price_data) * baseline_demand

estimated_counterfactual_units = (
    estimated_baseline_units * demand_factor
)

baseline_revenue = (
    baseline_price * estimated_baseline_units
)

counterfactual_revenue = (
    new_price * estimated_counterfactual_units
)

revenue_difference = (
    counterfactual_revenue - baseline_revenue
)

revenue_change = (
    revenue_difference / baseline_revenue
) * 100

# Display results
st.subheader("Revenue Comparison")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Estimated Baseline Revenue",
    f"₹{baseline_revenue:,.0f}"
)

col2.metric(
    "Counterfactual Revenue",
    f"₹{counterfactual_revenue:,.0f}"
)

col3.metric(
    "Revenue Change",
    f"{revenue_change:.2f}%"
)

st.write(
    "Revenue Difference:",
    f"₹{revenue_difference:,.2f}"
)
st.subheader("Baseline vs Counterfactual Revenue Trajectory")

import plotly.graph_objects as go

revenue_chart = go.Figure()

revenue_chart.add_trace(
    go.Scatter(
        x=df.index,
        y=price_data * baseline_demand,
        mode="lines",
        name="Baseline Revenue"
    )
)

revenue_chart.add_trace(
    go.Scatter(
        x=df.index,
        y=price_data * demand_factor * baseline_demand,
        mode="lines",
        name="Counterfactual Revenue"
    )
)

revenue_chart.update_layout(
    title="Estimated Revenue Comparison",
    xaxis_title="Property Records",
    yaxis_title="Estimated Revenue (INR)",
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(revenue_chart, use_container_width=True)
st.divider()

st.subheader("Download Pricing Analysis Report")

# Prepare downloadable results
results = pd.DataFrame({
    "Metric": [
        "Baseline Price (INR)",
        "Simulated Price (INR)",
        "Price Change (%)",
        "Demand Elasticity",
        "Baseline Estimated Revenue (INR)",
        "Counterfactual Revenue (INR)",
        "Revenue Difference (INR)",
        "Revenue Change (%)"
    ],
    "Value": [
        baseline_price,
        new_price,
        price_change,
        elasticity,
        baseline_revenue,
        counterfactual_revenue,
        revenue_difference,
        revenue_change
    ]
})

csv_data = results.to_csv(index=False)

st.download_button(
    label="Download Pricing Analysis (CSV)",
    data=csv_data,
    file_name="pricing_analysis_report.csv",
    mime="text/csv"
)
st.pyplot(fig)

with st.expander("Show data"):
    st.dataframe(df)
