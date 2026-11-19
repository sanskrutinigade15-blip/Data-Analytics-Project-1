import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Causal Impact Visualization", layout="wide")
st.title("Causal Impact Visualization")
st.write(
    "Estimates what would have happened **without** an intervention, using "
    "control series from the period before it, and compares that to what "
    "actually happened."
)

CSV_PATH = Path(__file__).parent / "causal_impact_data.csv"


def make_sample_data(n=120, intervention=90, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    x1 = 100 + np.cumsum(rng.normal(0, 1, n))
    x2 = 50 + np.cumsum(rng.normal(0, 0.7, n))
    y = 20 + 0.6 * x1 + 0.4 * x2 + rng.normal(0, 1.5, n)
    y[intervention:] += 12  # simulated effect
    return pd.DataFrame({"date": dates, "y": y, "x1": x1, "x2": x2})


# ---------- Load data ----------
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type="csv")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.sidebar.success("Using uploaded file")
elif CSV_PATH.is_file():
    df = pd.read_csv(CSV_PATH)
    st.sidebar.success("Using causal_impact_data.csv")
else:
    df = make_sample_data()
    st.sidebar.info("No CSV found, using sample data")
    df["Observation Date"] = pd.to_datetime(df["Observation Date"])
df = (
    df.groupby("Observation Date", as_index=False)
      .mean(numeric_only=True)
      .sort_values("Observation Date")
)

st.write("Available Dataset Columns:")
st.write(df.columns.tolist())
st.write("Dataset Shape:", df.shape)
st.write("First 5 Rows:")
st.dataframe(df.head())
st.divider()

st.header("What-if Pricing Simulator")

st.write(
    "Simulate how changes in property prices "
    "may affect estimated demand and revenue."
)

# ---------- Choose columns ----------
num_cols = df.select_dtypes("number").columns.tolist()
if len(num_cols) < 2:
    st.error("Need at least 2 numeric columns (a target and one control).")
    st.stop()

other_cols = [c for c in df.columns if c not in num_cols]
date_options = ["(use row number)"] + df.columns.tolist()
default_date = other_cols[0] if other_cols else date_options[0]
date_col = st.sidebar.selectbox(
    "Date column", date_options, index=date_options.index(default_date)
)
target = st.sidebar.selectbox("Target (what you measure)", num_cols, index=0)
control_choices = [c for c in num_cols if c != target]
controls = st.sidebar.multiselect(
    "Control columns", control_choices, default=control_choices
)

if date_col != "(use row number)":
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    x_axis = df[date_col]
else:
    x_axis = pd.Series(np.arange(len(df)))

df = df.dropna(subset=[target] + controls).reset_index(drop=True)
x_axis = x_axis.iloc[: len(df)].reset_index(drop=True)

if not controls:
    st.warning("Select at least one control column in the sidebar.")
    st.stop()
if len(df) < 20:
    st.error("Need at least 20 rows of data.")
    st.stop()

# ---------- Intervention point ----------
k = st.sidebar.slider(
    "Intervention starts at row", 10, len(df) - 2, int(len(df) * 0.75)
)
st.sidebar.caption(f"Intervention at: {x_axis.iloc[k]}")

# ---------- Model: regression on pre-period ----------
y = df[target].to_numpy(dtype=float)
X = np.column_stack([np.ones(len(df)), df[controls].to_numpy(dtype=float)])

beta, *_ = np.linalg.lstsq(X[:k], y[:k], rcond=None)
pred = X @ beta
resid = y[:k] - pred[:k]
dof = max(k - X.shape[1], 1)
sigma = np.sqrt(np.sum(resid**2) / dof)
lower, upper = pred - 1.96 * sigma, pred + 1.96 * sigma

effect = y - pred
post_effect = effect[k:]
avg_effect = post_effect.mean()
cum_effect = post_effect.sum()
rel_effect = 100 * post_effect.sum() / pred[k:].sum()
significant = not (
    (lower[k:] <= y[k:]).all() and (y[k:] <= upper[k:]).all()
) or abs(avg_effect) > 1.96 * sigma

# ---------- Summary ----------
c1, c2, c3 = st.columns(3)
c1.metric("Average effect", f"{avg_effect:,.2f}")
c2.metric("Cumulative effect", f"{cum_effect:,.2f}")
c3.metric("Relative effect", f"{rel_effect:,.1f}%")

if abs(avg_effect) > 1.96 * sigma:
    st.success("The effect looks statistically meaningful (beyond the pre-period noise).")
else:
    st.info("The effect is within normal pre-period noise, so it may not be real.")

# ---------- Plots ----------
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

axes[0].plot(x_axis, y, label="Observed", color="black")
axes[0].plot(x_axis, pred, "--", label="Predicted (no intervention)", color="tab:blue")
axes[0].fill_between(x_axis, lower, upper, color="tab:blue", alpha=0.2)
axes[0].set_title("Observed vs. counterfactual")
axes[0].legend()

axes[1].plot(x_axis, effect, color="tab:green")
axes[1].axhline(0, color="gray", linestyle=":")
axes[1].fill_between(x_axis, lower - pred, upper - pred, color="tab:green", alpha=0.2)
axes[1].set_title("Pointwise effect")

cum = np.concatenate([np.zeros(k), np.cumsum(post_effect)])
axes[2].plot(x_axis, cum, color="tab:red")
axes[2].axhline(0, color="gray", linestyle=":")
axes[2].set_title("Cumulative effect")

for ax in axes:
    ax.axvline(x_axis.iloc[k], color="red", linestyle="--", alpha=0.7)

fig.tight_layout()
st.pyplot(fig)

with st.expander("Show data"):
    st.dataframe(df)
