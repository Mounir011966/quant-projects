# Quant Projects

Academic-inspired personal project, building on stochastic calculus & financial derivatives courses (KTH / Centrale Méditerranée).

This repository contains compact implementations in **quantitative finance**, with a focus on option pricing and time series econometrics.

## Contents
- `notebooks/blackscholes.ipynb` – European option pricing (Black–Scholes closed form)  
- `notebooks/montecarlo.ipynb` – Monte Carlo pricer (European / exotic-ready)  
- `notebooks/spread.ipynb` – Cointegration tests, spread modeling and simple backtesting  
- `data/price_data.csv` – Sample dataset  
- `requirements.txt` – Python dependencies  

## Installation
```bash
git clone https://github.com/Mounir011966/quant-projects.git
cd quant-projects
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
Dependencies: numpy, pandas, scipy, matplotlib, statsmodels, yfinance, jupyter.


Run blackscholes.ipynb to compute closed-form option prices.

Run montecarlo.ipynb to simulate option pricing with Monte Carlo methods.

Run spread.ipynb for time series analysis, cointegration tests and backtesting.

## References
Black & Scholes (1973)

Hull, Options, Futures and Other Derivatives

Shreve, Stochastic Calculus for Finance II

Hamilton, Time Series Analysis

License: MIT
Author: Mounir ZEBBAR



---
