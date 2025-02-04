import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FakeCryptoDataGenerator:
    def __init__(self, symbol="XRP-USD", years=10):
        self.symbol = symbol
        self.years = years
        self.start_date = datetime.now() - timedelta(days=years * 365)
        self.end_date = datetime.now()
        self.data = None
        self.DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    def generate_data(self):
        """Generate fake historical data with realistic volatility"""
        try:
            # Generate date range
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq="D")
            np.random.seed(42)
            start_price = 0.20
            end_price = 20.00
            min_price = 0.20
            volatility = 0.03

            # Calculate the base trend to reach target price
            total_days = len(dates)
            base_trend = np.exp(
                np.linspace(np.log(start_price), np.log(end_price), total_days)
            )

            # Generate prices with high volatility
            prices = np.zeros(len(dates))
            prices[0] = start_price

            # Add multiple cyclical components for more complex patterns
            for i in range(1, len(dates)):
                # Multiple random variations
                daily_volatility = volatility * (
                    1 + 0.5 * np.sin(2 * np.pi * i / (365 * 2))
                )
                random_factor = np.random.normal(0, daily_volatility)

                # Multiple cyclical components
                cycle1 = 0.1 * np.sin(2 * np.pi * i / 365)  # Annual cycle
                cycle2 = 0.05 * np.sin(2 * np.pi * i / (365 * 2))  # Biannual cycle
                cycle3 = 0.03 * np.sin(2 * np.pi * i / (90))  # Quarterly cycle

                # Combine trend with randomness and cycles
                prices[i] = base_trend[i] * (
                    1 + random_factor + cycle1 + cycle2 + cycle3
                )

                # Add occasional larger moves
                if np.random.random() < 0.005:  # 0.5% chance of a significant move
                    prices[i] *= 1 + np.random.choice([-0.15, 0.15])  # 15% moves

                # Ensure price doesn't go below minimum
                prices[i] = max(prices[i], min_price)

            # Add some market correction periods
            for i in range(total_days // 500):  # Every ~500 days
                correction_start = np.random.randint(0, total_days - 30)
                correction_length = np.random.randint(10, 30)
                correction_size = np.random.uniform(0.1, 0.3)  # 10-30% correction
                prices[correction_start : correction_start + correction_length] *= (
                    1 - correction_size
                )

            # Round prices to 2 decimal places
            prices = np.round(prices, 2)

            # Generate related prices with realistic spreads
            spread_factor = 0.005  # 0.5% spread
            open_prices = np.round(
                prices * (1 + np.random.normal(0, spread_factor, len(dates))), 2
            )
            high_prices = np.round(
                np.maximum(prices, open_prices)
                * (1 + np.abs(np.random.normal(0, spread_factor, len(dates)))),
                2,
            )
            low_prices = np.round(
                np.minimum(prices, open_prices)
                * (1 - np.abs(np.random.normal(0, spread_factor, len(dates)))),
                2,
            )

            # Ensure high is always highest and low is always lowest
            high_prices = np.maximum.reduce([high_prices, prices, open_prices])
            low_prices = np.minimum.reduce([low_prices, prices, open_prices])

            # Generate realistic volume with log-normal distribution
            mean_volume = 50000000  # 50M average volume
            volume_std = 0.5
            volumes = np.round(
                np.random.lognormal(np.log(mean_volume), volume_std, len(dates))
            )

            # Create DataFrame
            self.data = pd.DataFrame(
                {
                    "Price": prices,
                    "Close": prices,
                    "High": high_prices,
                    "Low": low_prices,
                    "Open": open_prices,
                    "Volume": volumes.astype(int),
                },
                index=dates,
            )
            self.data.index.name = "Date"

            # Add some basic statistics logging
            logger.info(f"Successfully generated {len(self.data)} records")
            logger.info(f"Price statistics:")
            logger.info(f"Mean price: ${self.data['Price'].mean():.2f}")
            logger.info(f"Max price: ${self.data['Price'].max():.2f}")
            logger.info(f"Min price: ${self.data['Price'].min():.2f}")
            logger.info(f"Price volatility: {self.data['Price'].std():.2f}")
            logger.info("\nData sample:")
            logger.info(f"\n{self.data.head()}")

            return True

        except Exception as e:
            logger.error(f"Error generating data: {str(e)}")
            logger.error("Full traceback:", exc_info=True)
            return False

    def save_data(self, filename="xrp_simulated_data.csv"):
        """Save the generated data to CSV"""
        try:
            if self.data is None:
                raise ValueError("No data to save. Run generate_data first.")

            # Create data directory if it doesn't exist
            os.makedirs(self.DATA_DIR, exist_ok=True)
            filepath = os.path.join(self.DATA_DIR, filename)

            # Save to CSV
            self.data.to_csv(filepath)

            logger.info(f"Successfully saved {len(self.data)} records to {filepath}")
            logger.info("\nSaved data sample:")
            logger.info(f"\n{pd.read_csv(filepath).head()}")

            return filepath

        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            logger.error("Full traceback:", exc_info=True)
            return None

    def verify_data(self, filepath):
        """Verify the saved data"""
        try:
            logger.info(f"Verifying data in {filepath}")

            # Read the saved file
            df = pd.read_csv(filepath)

            # Basic checks
            if df.empty:
                raise ValueError("Saved data is empty")

            expected_columns = [
                "Date",
                "Price",
                "Close",
                "High",
                "Low",
                "Open",
                "Volume",
            ]
            missing_columns = set(expected_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing columns: {missing_columns}")

            logger.info("Data verification passed")
            logger.info("\nVerified data sample:")
            logger.info(f"\n{df.head()}")

            return True

        except Exception as e:
            logger.error(f"Data verification failed: {str(e)}")
            logger.error("Full traceback:", exc_info=True)
            return False

    def plot_data(self):
        """Plot the generated price data"""
        try:
            if self.data is None:
                raise ValueError("No data to plot. Run generate_data first.")

            plt.figure(figsize=(15, 7))
            plt.plot(self.data.index, self.data["Price"], label="Price")
            plt.title(f"{self.symbol} Simulated Price History")
            plt.xlabel("Date")
            plt.ylabel("Price ($)")
            plt.grid(True)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save the plot
            plt.savefig(self.DATA_DIR + "/price_plot.png")
            plt.close()

            logger.info(f"Price plot saved as '${self.DATA_DIR}/price_plot.png'")
            return True

        except Exception as e:
            logger.error(f"Error plotting data: {str(e)}")
            return False


def main():
    try:
        # Initialize generator
        generator = FakeCryptoDataGenerator()

        # Generate data
        if not generator.generate_data():
            logger.error("Failed to generate data. Exiting...")
            return

        # Save data
        filepath = generator.save_data()
        if not filepath:
            logger.error("Failed to save data")
            return

        # Verify saved data
        if not generator.verify_data(filepath):
            logger.error("Data verification failed")
            return

        # Plot data
        if not generator.plot_data():
            logger.error("Failed to plot data")
            return

        logger.info(
            f"Data generation completed successfully. File saved at: {filepath}"
        )

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
