# CashFlow Intelligence

An explainable liquidity monitor for small businesses. Import signed cash transactions, inspect weekly history, and run a reproducible moving-block bootstrap to estimate future cash ranges and breach risk.

## What it does

- Validates a CSV containing `date,amount` (inflows positive, outflows negative).
- Forecasts weekly net cash movement with a circular moving-block bootstrap. Resampling short blocks preserves local dependence better than sampling individual weeks independently.
- Shows 10th, 50th and 90th percentile cash paths, probability of crossing a user-selected minimum cash threshold, and the median forecast low point.
- Uses a fixed random seed for repeatability and states the stationary-history assumption.

This is a decision-support prototype, not an accounting system or financial advice. It does not ingest bank credentials or claim that synthetic example data represents a real company. The sample is deterministic synthetic data; upload your own export to analyze actual history.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Use `data/sample_transactions.csv` to reproduce the demo. The line and uncertainty-band composition is adapted from AnyPlot's [Plotly confidence-interval chart](https://anyplot.ai/line-confidence/python/plotly) (MIT) and rebuilt around this project's empirical forecast data. AnyPlot publishes runnable chart source and license metadata in its [catalog](https://anyplot.ai/).

## Input contract

`date` must parse as a calendar date. `amount` must be a finite number in one consistent currency. Positive amounts are receipts; negative amounts are payments. Keep one currency per file and exclude transfers between accounts unless they change the consolidated cash position. Enter the cash balance at the end of the final observed week; the history is reconstructed backward from that anchor. At least 12 consecutive weeks with observations are required. Missing weeks are treated as zero net movement; this assumption is visible in the historical chart.

## Method and limits

The model resamples circular blocks of historical weekly net flows, adds those changes to the current cash balance, and repeats across the chosen horizon. This preserves short-run dependence while making few parametric assumptions. It assumes recent weekly flows are representative of the near future. Structural changes, seasonality, one-off invoices, payment terms, tax dates, and known future commitments can make this assumption wrong. The dashboard exposes percentile ranges; they are empirical simulation quantiles, not calibrated guarantees. Check results against a simple base case and add known future events before relying on the forecast.

## Quality checks

```bash
python -m pytest
```

Tests cover input validation, deterministic forecasts, quantile ordering, and breach-risk bounds.

## Data provenance

`data/sample_transactions.csv` contains invented transactions for a fictional business. It is labeled synthetic and contains no customer or bank data. Replace it with a properly anonymized export for real analysis.

## License

MIT. See `LICENSE`.
