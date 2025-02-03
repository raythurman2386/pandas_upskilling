import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import ta
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.utils.logger import setup_logger


class CryptoTradePredictor:
    def __init__(self, csv_path):
        self.logger = setup_logger("CryptoTradePredictor")
        self.csv_path = Path(csv_path)
        self.data = None
        self.features = None
        self.targets = None
        self.models = {}

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

    def load_data(self):
        """Load and prepare the data"""
        try:
            self.logger.info("Loading data from CSV...")

            # Load the CSV file
            self.data = pd.read_csv(
                self.csv_path, parse_dates=["Date"], index_col="Date"
            )

            # Verify required columns
            required_columns = {"Price", "Close", "High", "Low", "Open", "Volume"}
            missing_columns = required_columns - set(self.data.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Convert Volume to integer and verify numeric columns
            self.data["Volume"] = self.data["Volume"].astype(int)
            numeric_columns = ["Price", "Close", "High", "Low", "Open"]
            self.data[numeric_columns] = self.data[numeric_columns].astype(float)

            self.logger.info(f"Successfully loaded {len(self.data)} records")
            self.logger.info(
                f"Date range: {self.data.index.min()} to {self.data.index.max()}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}", exc_info=True)
            return False

    def add_technical_indicators(self):
        """Add technical indicators to the dataset"""
        try:
            self.logger.info("Adding technical indicators...")

            # Trend Indicators
            self.data["SMA_20"] = ta.trend.sma_indicator(self.data["Close"], window=20)
            self.data["SMA_50"] = ta.trend.sma_indicator(self.data["Close"], window=50)
            self.data["EMA_20"] = ta.trend.ema_indicator(self.data["Close"], window=20)
            self.data["MACD"] = ta.trend.macd_diff(self.data["Close"])

            # Momentum Indicators
            self.data["RSI"] = ta.momentum.rsi(self.data["Close"])
            self.data["Stoch"] = ta.momentum.stoch(
                self.data["High"], self.data["Low"], self.data["Close"]
            )
            self.data["Williams_R"] = ta.momentum.williams_r(
                self.data["High"], self.data["Low"], self.data["Close"]
            )

            # Volatility Indicators
            bb = ta.volatility.BollingerBands(self.data["Close"])
            self.data["BB_upper"] = bb.bollinger_hband()
            self.data["BB_lower"] = bb.bollinger_lband()
            self.data["ATR"] = ta.volatility.average_true_range(
                self.data["High"], self.data["Low"], self.data["Close"]
            )

            # Volume Indicators
            self.data["OBV"] = ta.volume.on_balance_volume(
                self.data["Close"], self.data["Volume"]
            )

            # Custom Features
            self.data["Price_Change"] = self.data["Close"].pct_change()
            self.data["Volume_Change"] = self.data["Volume"].pct_change()
            self.data["Volatility"] = self.data["Close"].rolling(window=20).std()
            self.data["Price_Range"] = (
                self.data["High"] - self.data["Low"]
            ) / self.data["Close"]

            # Drop NaN values
            self.data.dropna(inplace=True)

            self.logger.info(f"Added {len(self.data.columns)} technical indicators")
            return True

        except Exception as e:
            self.logger.error(
                f"Error adding technical indicators: {str(e)}", exc_info=True
            )
            return False

    def create_labels(self, forward_days=5, threshold=0.02):
        """Create trading signals based on future price movements"""
        try:
            self.logger.info(
                f"Creating trading signals (forward_days={forward_days}, threshold={threshold})"
            )

            # Calculate future returns
            future_returns = (
                self.data["Close"].shift(-forward_days) / self.data["Close"] - 1
            )

            # Create labels: 2 (buy), 0 (sell), 1 (hold)
            # Modified to use non-negative integers for XGBoost compatibility
            self.data["Target"] = np.where(
                future_returns > threshold,
                2,
                np.where(future_returns < -threshold, 0, 1),
            )

            # Remove last N rows where we don't have future data
            self.data = self.data[:-forward_days]

            label_distribution = self.data["Target"].value_counts()
            self.logger.info(f"Label distribution:\n{label_distribution}")

            # Store label mapping for later use
            self.label_mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}
            return True

        except Exception as e:
            self.logger.error(f"Error creating labels: {str(e)}", exc_info=True)
            return False

    def prepare_features(self):
        """Prepare features for modeling"""
        try:
            # Select features (exclude price data and target)
            exclude_columns = [
                "Target",
                "Price",
                "Close",
                "High",
                "Low",
                "Open",
                "Volume",
            ]
            feature_columns = [
                col for col in self.data.columns if col not in exclude_columns
            ]

            self.features = self.data[feature_columns]
            self.targets = self.data["Target"]

            # Scale features
            scaler = StandardScaler()
            self.features = pd.DataFrame(
                scaler.fit_transform(self.features),
                columns=self.features.columns,
                index=self.features.index,
            )

            self.logger.info(f"Prepared {len(feature_columns)} features for modeling")
            return True

        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}", exc_info=True)
            return False

    def train_models(self):
        """Train and evaluate models"""
        try:
            # Create train/validation split for early stopping
            X_train, X_val, y_train, y_val = train_test_split(
                self.features,
                self.targets,
                test_size=0.2,
                random_state=42,
                shuffle=False  # Maintain temporal order
            )

            # Initialize models with updated configurations
            models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1  # Use all available cores
                ),
                'xgboost': xgb.XGBClassifier(
                    n_estimators=1000,
                    max_depth=6,
                    learning_rate=0.1,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='mlogloss',
                    objective='multi:softproba',
                    num_class=3,
                    tree_method='hist',
                    early_stopping_rounds=50
                )
            }

            # Train and evaluate each model
            for name, model in models.items():
                self.logger.info(f"Training {name}...")

                if name == 'xgboost':
                    # Train with early stopping
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                    self.logger.info(f"Best iteration: {model.best_iteration}")
                else:
                    model.fit(X_train, y_train)

                self.models[name] = model

                # Evaluate on validation set
                predictions = model.predict(X_val)
                accuracy = accuracy_score(y_val, predictions)
                self.logger.info(f"{name} validation accuracy: {accuracy:.4f}")

                # Print classification report
                report = classification_report(
                    y_val,
                    predictions,
                    target_names=['SELL', 'HOLD', 'BUY']
                )
                print(f"\nValidation Classification Report for {name}:")
                print(report)

            return True

        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}", exc_info=True)
            return False

    def generate_signals(self, confidence_threshold=0.6):
        """Generate trading signals with confidence scores"""
        try:
            signals = pd.DataFrame(index=self.features.index)

            for name, model in self.models.items():
                proba = model.predict_proba(self.features)
                signals[f"{name}_confidence"] = proba.max(axis=1)
                predictions = model.predict(self.features)
                signals[f"{name}_prediction"] = predictions

            # Generate consensus signal
            prediction_cols = [col for col in signals.columns if "prediction" in col]
            confidence_cols = [col for col in signals.columns if "confidence" in col]

            signals["consensus"] = signals[prediction_cols].mode(axis=1)[0]
            signals["avg_confidence"] = signals[confidence_cols].mean(axis=1)

            # Final trading signal using label mapping
            signals["signal"] = signals["consensus"].map(self.label_mapping)

            # Filter by confidence threshold
            signals.loc[signals["avg_confidence"] < confidence_threshold, "signal"] = (
                "HOLD"
            )

            # Add price data for reference
            signals["price"] = self.data["Close"]

            # Add additional metrics
            signals["future_return"] = (
                self.data["Close"].pct_change(5).shift(-5)
            )  # 5-day future return
            signals["rsi"] = self.data["RSI"]
            signals["volume_change"] = self.data["Volume_Change"]

            self.logger.info("\nSignal distribution:")
            self.logger.info(signals["signal"].value_counts())

            return signals

        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}", exc_info=True)
            return None


def main():
    try:
        # Initialize and run the predictor
        predictor = CryptoTradePredictor("xrp_historical_data.csv")

        if not all(
            [
                predictor.load_data(),
                predictor.add_technical_indicators(),
                predictor.create_labels(),
                predictor.prepare_features(),
                predictor.train_models(),
            ]
        ):
            raise Exception("Pipeline execution failed")

        # Generate and save trading signals
        signals = predictor.generate_signals()
        if signals is not None:
            # Save detailed signals
            signals.to_csv("trading_signals.csv")

            # Display recent signals with additional metrics
            print("\nRecent Trading Signals:")
            recent_signals = signals.tail(10)[
                ["signal", "avg_confidence", "price", "rsi", "volume_change"]
            ].round(4)
            print(recent_signals.to_string())

            # Display signal statistics
            print("\nSignal Distribution:")
            print(signals["signal"].value_counts())

            print("\nAverage Confidence by Signal:")
            print(signals.groupby("signal")["avg_confidence"].mean().round(4))

        else:
            print("Failed to generate trading signals")

    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
