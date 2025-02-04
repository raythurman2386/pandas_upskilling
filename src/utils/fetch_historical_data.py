import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class CryptoDataFetcher:
    def __init__(self, symbol="XRP-USD", years=10):
        self.symbol = symbol
        self.years = years
        self.start_date = datetime.now() - timedelta(days=years * 365)
        self.end_date = datetime.now()
        self.data = None

    def fetch_data(self):
        """Fetch historical data from yfinance"""
        try:
            logger.info(
                f"Fetching {self.symbol} data from {self.start_date.date()} to {self.end_date.date()}"
            )

            self.data = yf.download(
                self.symbol,
                start=self.start_date,
                end=self.end_date,
                interval="1d",
                progress=False,
            )

            if self.data.empty:
                raise ValueError("No data retrieved from yfinance")

            logger.info(f"Successfully fetched {len(self.data)} records")
            logger.info(f"Retrieved columns: {self.data.columns.tolist()}")
            logger.info("\nRaw data sample:")
            logger.info(f"\n{self.data.head()}")
            logger.info(f"\nIndex type: {type(self.data.index)}")
            logger.info(
                f"Date range: {self.data.index.min()} to {self.data.index.max()}"
            )

            return True

        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            logger.error("Full traceback:", exc_info=True)
            return False

    def process_data(self):
        """Process and clean the data"""
        try:
            if self.data is None:
                raise ValueError("No data to process. Run fetch_data first.")

            logger.info(f"Original columns: {self.data.columns.tolist()}")
            logger.info("\nOriginal data sample:")
            logger.info(f"\n{self.data.head()}")

            # Create a clean DataFrame with the correct structure
            processed_data = pd.DataFrame()

            # Copy data with proper column names
            processed_data["Price"] = self.data["Close"]
            processed_data["Close"] = self.data["Close"]
            processed_data["High"] = self.data["High"]
            processed_data["Low"] = self.data["Low"]
            processed_data["Open"] = self.data["Open"]
            processed_data["Volume"] = self.data["Volume"]

            # Set the index
            processed_data.index = self.data.index
            processed_data.index.name = "Date"

            # Round numeric columns to 8 decimal places
            numeric_columns = ["Price", "Close", "High", "Low", "Open"]
            processed_data[numeric_columns] = processed_data[numeric_columns].round(8)

            # Convert volume to integer
            processed_data["Volume"] = processed_data["Volume"].astype(int)

            # Replace the original data with processed data
            self.data = processed_data

            logger.info("\nProcessed data sample:")
            logger.info(f"\n{self.data.head()}")
            logger.info(f"\nFinal columns: {self.data.columns.tolist()}")
            logger.info(f"Data shape: {self.data.shape}")

            return True

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            logger.error("Full traceback:", exc_info=True)
            return False

    def save_data(self, filename="xrp_historical_data.csv"):
        """Save the processed data to CSV"""
        try:
            if self.data is None:
                raise ValueError(
                    "No data to save. Run fetch_data and process_data first."
                )

            # Create data directory if it doesn't exist
            os.makedirs("data", exist_ok=True)
            filepath = os.path.join("data", filename)

            # Save to CSV with date index
            self.data.to_csv(filepath)

            # Verify the saved file by reading it back
            saved_data = pd.read_csv(filepath)

            logger.info(f"Successfully saved {len(self.data)} records to {filepath}")
            logger.info("\nSaved data sample:")
            logger.info(f"\n{saved_data.head()}")
            logger.info(f"\nSaved columns: {saved_data.columns.tolist()}")

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

            # Convert date column
            df["Date"] = pd.to_datetime(df["Date"])

            # Check data types
            numeric_columns = ["Price", "Close", "High", "Low", "Open"]
            for col in numeric_columns:
                if not pd.api.types.is_float_dtype(df[col]):
                    raise ValueError(f"Column {col} is not float type")

            if not pd.api.types.is_integer_dtype(df["Volume"]):
                df["Volume"] = df["Volume"].astype(int)

            logger.info("Data verification passed")
            logger.info("\nVerified data sample:")
            logger.info(f"\n{df.head()}")

            return True

        except Exception as e:
            logger.error(f"Data verification failed: {str(e)}")
            logger.error("Full traceback:", exc_info=True)
            return False


def main():
    try:
        # Initialize fetcher
        fetcher = CryptoDataFetcher()

        # Fetch data
        if not fetcher.fetch_data():
            logger.error("Failed to fetch data. Exiting...")
            return

        # Process data
        if not fetcher.process_data():
            logger.error("Failed to process data. Exiting...")
            return

        # Save data
        filepath = fetcher.save_data()
        if not filepath:
            logger.error("Failed to save data")
            return

        # Verify saved data
        if not fetcher.verify_data(filepath):
            logger.error("Data verification failed")
            return

        logger.info(f"Data pipeline completed successfully. File saved at: {filepath}")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
