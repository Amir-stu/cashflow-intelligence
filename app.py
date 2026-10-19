from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecast import DataValidationError, forecast_cash, validate_transactions


st.set_page_config(page_title="CashFlow Intelligence", page_icon="💧", layout="wide")
st.title("CashFlow Intelligence")
st.caption("A transparent cash runway forecast for small businesses. Use your own data or the clearly labeled synthetic demo.")

uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])
if uploaded:
    raw = pd.read_csv(uploaded)
    source_label = uploaded.name
else:
    raw = pd.read_csv(Path(__file__).parent / "data" / "sample_transactions.csv")
    source_label = "Synthetic demo data"

try:
    tx = validate_transactions(raw)
except (DataValidationError, pd.errors.ParserError) as exc:
    st.error(str(exc))
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    starting_cash = st.number_input("Cash balance at end of final history week", min_value=0.0, value=85_000.0, step=5_000.0)
with col2:
    threshold = st.number_input("Minimum cash threshold", value=20_000.0, step=5_000.0)
with col3:
    horizon = st.slider("Forecast horizon (weeks)", min_value=4, max_value=26, value=13)
currency = st.text_input("Currency code (ISO 4217)", value="EUR", max_chars=3).strip().upper()
if len(currency) != 3 or not currency.isalpha():
    st.error("Enter a three-letter currency code, such as EUR or USD.")
    st.stop()

st.caption(f"Data source: {source_label} · {len(tx):,} transactions · {tx['date'].min():%d %b %Y} to {tx['date'].max():%d %b %Y}")
try:
    result = forecast_cash(tx, starting_cash, horizon, threshold)
except DataValidationError as exc:
    st.error(str(exc))
    st.stop()

metrics = st.columns(3)
metrics[0].metric("Median cash at horizon", f"{currency} {result.weekly_cash.iloc[-1]['p50']:,.0f}")
metrics[1].metric("Chance cash falls below threshold", f"{result.probability_below_threshold:.0%}")
metrics[2].metric("Median lowest cash in forecast", f"{currency} {result.median_minimum_cash:,.0f}")

weekly_history = tx.set_index("date")["amount"].resample("W-SUN").sum().asfreq("W-SUN", fill_value=0.0)
reconstructed_history = starting_cash - weekly_history.iloc[::-1].cumsum().iloc[::-1] + weekly_history
fig = go.Figure()
fig.add_trace(go.Scatter(x=weekly_history.index, y=reconstructed_history, mode="lines", line={"color": "#4467A3", "width": 2}, name="Reconstructed history"))
# Adapted from AnyPlot's MIT Plotly confidence-band pattern: upper bound, filled lower bound, median line.
fig.add_trace(go.Scatter(x=result.weekly_cash["date"], y=result.weekly_cash["p90"], line={"width": 0}, showlegend=False, name="90th percentile"))
fig.add_trace(go.Scatter(x=result.weekly_cash["date"], y=result.weekly_cash["p10"], fill="tonexty", fillcolor="rgba(68,103,163,0.18)", line={"width": 0}, name="10–90% range"))
fig.add_trace(go.Scatter(x=result.weekly_cash["date"], y=result.weekly_cash["p50"], mode="lines", line={"color": "#009E73", "width": 3}, name="Median forecast"))
fig.add_vline(x=weekly_history.index[-1], line_dash="dot", line_color="#6B6A63", annotation_text="Forecast starts")
fig.add_hline(y=threshold, line_dash="dash", line_color="#AE3030", annotation_text="Minimum cash threshold")
fig.update_layout(title="Cash history and projected balance · empirical simulation range", xaxis_title="Week ending", yaxis_title=f"Cash balance ({currency})", hovermode="x unified", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

with st.expander("Method, assumptions, and limitations"):
    st.write("Historical signed amounts are aggregated into Sunday-ending weeks. The simulator samples circular four-week blocks, preserving short-range dependence, then adds the changes to current cash. Percentiles are simulation quantiles, not calibrated guarantees.")
    st.write("The forecast assumes past weekly cash movements remain representative. It does not know about seasonality, one-off contracts, payment delays, taxes, or future commitments. Add those before using this as a decision input. This tool is not financial advice.")
