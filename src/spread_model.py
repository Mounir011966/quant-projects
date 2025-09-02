from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

def cointegration(y1,y2):
    
    y2=sm.add_constant(y2)
    model = sm.OLS(y1, y2).fit()
    residuals = model.resid
    
    adf_result = adfuller(residuals)
    p_value = adf_result[1]
    score = adf_result[0]
    
    return p_value, score 

df = pd.read_csv("data/price_data.csv", index_col=0, parse_dates=True)

y1 = df["XOM"]
y2 = df["CVX"]
pvalue, score = cointegration(y1,y2)

scaler = StandardScaler()
y1_scaled = scaler.fit_transform(y1.values.reshape(-1, 1))
y2_scaled = scaler.fit_transform(y2.values.reshape(-1, 1))


spread = (y1_scaled - y2_scaled)

Z = (spread - np.mean(spread)) / np.std(spread)

positions = []

positions = np.where(
    Z > 1,     # condition 1
    -1,              # valeur si condition 1 vraie
    np.where(
        Z < -1,  # condition 2 (si condition 1 est fausse)
        1,             # valeur si condition 2 vraie
        0              # valeur si condition 2 fausse
    )
)
positions = positions.flatten()

returns = np.diff(positions)
# Appliquer les positions avec un décalage pour éviter le lookahead bias
strategy_returns = returns[:-1] * positions[1:-1]

# Cumul des gains
cumulative_returns = np.cumsum(strategy_returns)

# Affichage
plt.figure(figsize=(14, 6))
plt.plot(df.index[2:], cumulative_returns, label="Strategy Cumulative Returns")
plt.axhline(0, color='gray', linestyle='--')
plt.title("Cumulative Returns of the Z-score Strategy")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#plt.figure(figsize=(14, 6))
#plt.plot(df.index, Z, label="Z-score du spread", color='blue')
#plt.axhline(0, color='black', linestyle='--')
#plt.axhline(1, color='red', linestyle='--', label="Seuil +1 (vendre)")
#plt.axhline(-1, color='green', linestyle='--', label="Seuil -1 (acheter)")
#plt.title(f"Z-score ")
#plt.xlabel("Date")
#plt.ylabel("Z-score")
#plt.legend()
#plt.grid(True)
#plt.show()
