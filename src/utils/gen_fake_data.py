import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate a date range (e.g., 10 years of daily data)
dates = pd.date_range(start="2014-01-01", end="2025-02-03", freq="D")

# Simulate XRP price (starting at $0.30, adding slight upward trend + noise)
prices = np.cumsum(np.random.normal(0.02, 0.1, len(dates))) + 0.30
prices = np.clip(prices, 0.10, None)  # Ensure no negative prices

# Simulate other stock metrics
volumes = np.random.randint(1000000, 100000000, len(dates))  # Random volume
open_prices = prices + np.random.normal(0.01, 0.05, len(dates))
high_prices = prices + np.random.uniform(0.02, 0.1, len(dates))
low_prices = prices - np.random.uniform(0.02, 0.1, len(dates))

# Create DataFrame
df = pd.DataFrame(
    {
        "Date": dates,
        "Open": open_prices,
        "High": high_prices,
        "Low": low_prices,
        "Close": prices,
        "Volume": volumes,
    }
)

# Save to CSV
csv_filename = "../xrp_simulated_data.csv"
df.to_csv(csv_filename, index=False)

print(f"Saved simulated XRP data to {csv_filename}")
