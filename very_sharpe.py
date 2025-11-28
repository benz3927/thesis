import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_gsw_yield(n, beta0, beta1, beta2, beta3, tau1, tau2):
    """Compute GSW yield at maturity n"""
    tau1 = 0.0001 if (tau1 == 0 or pd.isna(tau1)) else tau1
    if tau2 in [-999.99, 0] or pd.isna(tau2):
        beta3 = 0  # Switch to Nelson-Siegel

    term1 = beta1 * (1 - np.exp(-n / tau1)) / (n / tau1)
    term2 = beta2 * ((1 - np.exp(-n / tau1)) / (n / tau1) - np.exp(-n / tau1))
    term3 = 0 if beta3 == 0 else beta3 * ((1 - np.exp(-n / tau2)) / (n / tau2) - np.exp(-n / tau2))

    return beta0 + term1 + term2 + term3


# --- Load & prepare data ---
df = pd.read_csv("feds200628.csv", skiprows=9)
df["Date"] = pd.to_datetime(df["Date"])
df = df[df["Date"].dt.year >= 1970]  # start from 2000

# Extract parameters
cols = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

# --- Compute yields at chosen maturities ---
def gsw_series(maturity):
    return df.apply(lambda row: calculate_gsw_yield(
        maturity, row.BETA0, row.BETA1, row.BETA2, row.BETA3, row.TAU1, row.TAU2
    ), axis=1)

y_3m = gsw_series(0.25)
y_6q = gsw_series(1.5)
y_7q = gsw_series(1.75)

# --- Compute spreads ---
spread_2_10 = df["SVENY10"] - df["SVENY02"]
fwd6 = 7 * y_7q - 6 * y_6q
ntfs = fwd6 - y_3m  # Near-term forward spread

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], spread_2_10, label="2–10 Spread (10Y - 2Y)", color="C0", lw=2)
plt.plot(df["Date"], ntfs, label="Near-Term Forward Spread", color="C1", lw=2)
plt.axhline(0, color="black", linestyle="--", linewidth=0.6)
plt.title("Figure 1. 2–10 vs. Near-Term Forward Spread (1970–Present)", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Percentage Points")
plt.legend()

plt.tight_layout()
plt.show()
