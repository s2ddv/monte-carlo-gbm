# Research report

## Status

This is a research outline, not an experimental results report. No backtest or
new simulation is claimed here.

## Current model

The implemented single-asset model estimates annualized drift and volatility
from adjusted daily closes, then simulates exact GBM increments. Terminal
statistics are compared with the analytical lognormal distribution. The CLI
reports probability of loss, VaR and CVaR for the selected horizon.

## Questions for future research

1. How stable are drift and volatility across calibration windows?
2. How well do predicted terminal quantiles and VaR cover held-out outcomes?
3. How sensitive are tail estimates to simulation count and model assumptions?

## Evidence needed

Use dated, frozen inputs and chronological train/test splits. Record data
source, observation dates, code revision, dependency versions, random seed,
metrics and uncertainty. Separate Monte Carlo sampling error from calibration
uncertainty and model error. Report results only after running and reviewing
the experiments.
