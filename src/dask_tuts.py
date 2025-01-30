import dask.dataframe as dd
import numpy as np
import pandas as pd
import geopandas as gpd
from dask.diagnostics import ProgressBar
from datetime import datetime, timedelta
import random
import string

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, classification_report, \
    accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

from src.config.logging_config import CURRENT_LOGGING_CONFIG
from src.utils.logger import setup_logger

logger = setup_logger(
    "dask_and_dfs",
    log_level=CURRENT_LOGGING_CONFIG["log_level"],
    log_dir=CURRENT_LOGGING_CONFIG["log_dir"],
)


def perform_dask_operations():
    logger.info("Starting Dask operations")
    logger.info("Creating a large DataFrame with datetime index")
    start_date = datetime(2025, 1, 30)
    dates = [start_date + timedelta(minutes=i) for i in range(1000000)]
    df = pd.DataFrame({
        'timestamp': dates,
        'id': range(1000000),
        'value': np.random.randn(1000000),
        'category': np.random.choice(['A', 'B', 'C'], 1000000)
    })
    df.set_index('timestamp', inplace=True)

    # Convert to Dask DataFrame
    logger.info("Converting to Dask DataFrame with 4 partitions")
    ddf = dd.from_pandas(df, npartitions=4)

    with ProgressBar():
        mean_value = ddf['value'].mean().compute()
        logger.info(f"Mean value: {mean_value}")

        grouped = ddf.groupby('category').agg({
            'id': 'count',
            'value': ['mean', 'std', 'min', 'max']
        }).compute()
        logger.info(f"Grouped data:\n{grouped}")

        # Resample only numeric columns
        numeric_columns = ['id', 'value']  # Specify numeric columns explicitly
        resampled = ddf[numeric_columns].resample('1h').mean().compute()
        logger.info(f"Resampled data (first 5 rows):\n{resampled.head()}")

    logger.info("Dask operations completed")


def generate_large_graph(num_nodes=50, max_connections=5):
    nodes = list(string.ascii_uppercase) + [f'A{i}' for i in range(num_nodes - 26)]
    graph = {node: [] for node in nodes}

    for node in nodes:
        num_connections = random.randint(0, min(max_connections, num_nodes - 1))
        possible_neighbors = [n for n in nodes if n != node and (n, 0) not in graph[node]]
        neighbors = random.sample(possible_neighbors, min(num_connections, len(possible_neighbors)))
        for neighbor in neighbors:
            distance = random.randint(1, 100)
            graph[node].append((neighbor, distance))

    return graph


def dfs_iterative(graph, start_node):
    visited = set()
    stack = [start_node]
    path = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            path.append(node)
            logger.debug(f"Visited: {node}")

            for neighbor, distance in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return path


def perform_graph_operations():
    logger.info("Starting graph operations")

    large_graph = generate_large_graph(50, 5)
    start_node = "A"
    result = dfs_iterative(large_graph, start_node)

    logger.info(f"DFS path from {start_node}: {' -> '.join(result)}")
    logger.info(f"Total nodes visited: {len(result)}")

    logger.info(f"Total nodes in graph: {len(large_graph)}")
    logger.info("Number of connections for each node:")
    for node, connections in large_graph.items():
        logger.info(f"{node}: {len(connections)}")

    logger.info("Graph operations completed")


def combined_dask_and_dfs():
    logger.info("Starting combined Dask and DFS operations")
    logger.info("Creating a large DataFrame with datetime index")
    start_date = datetime(2025, 1, 30)
    dates = [start_date + timedelta(minutes=i) for i in range(1000000)]
    df = pd.DataFrame({
        'timestamp': dates,
        'id': range(1000000),
        'value': np.random.randn(1000000),
        'category': np.random.choice(['A', 'B', 'C'], 1000000)
    })
    df.set_index('timestamp', inplace=True)
    ddf = dd.from_pandas(df, npartitions=4)

    # Use Dask to calculate mean value
    with ProgressBar():
        mean_value = ddf['value'].mean().compute()
    logger.info(f"Mean value from Dask: {mean_value}")

    graph = generate_large_graph(50, 5)
    start_node = "A"
    dfs_result = dfs_iterative(graph, start_node)

    # Use Dask to filter data based on DFS result
    with ProgressBar():
        filtered_data = ddf[ddf['category'].isin(dfs_result[:10])].compute()

    logger.info(f"Filtered data based on DFS (first 5 rows):\n{filtered_data.head()}")
    logger.info("Combined Dask and DFS operations completed")


def perform_ml_task():
    logger.info("Starting Machine Learning task")

    # Generate some example flowline data
    np.random.seed(42)
    n_samples = 10000
    X = np.random.rand(n_samples, 3)  # 3 features for testing
    y = 2 * X[:, 0] + 0.5 * X[:, 1] - 1.5 * X[:, 2] + np.random.normal(0, 0.1, n_samples)  # Target: efficiency

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info("***Model Performance:***")
    logger.info(f"Mean Squared Error: {mse}")
    logger.info(f"R-squared Score: {r2}")

    # Feature importance
    feature_importance = model.feature_importances_
    for i, importance in enumerate(feature_importance):
        logger.info(f"Feature {i+1} importance: {importance}")

    logger.info("Machine Learning task completed")


def expanded_ml_task():
    logger.info("Starting Expanded Machine Learning task")
    np.random.seed(42)
    n_samples = 10000
    X = np.random.rand(n_samples, 3)
    y = 2 * X[:, 0] + 0.5 * X[:, 1] - 1.5 * X[:, 2] + np.random.normal(0, 0.1, n_samples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=42))
    ])

    # Hyperparameter tuning
    param_grid = {
        'rf__n_estimators': [50, 100, 200],
        'rf__max_depth': [None, 10, 20]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    logger.info(f"Best parameters: {grid_search.best_params_}")

    y_pred = grid_search.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info("***Model Performance:***")
    logger.info(f"Mean Squared Error: {mse}")
    logger.info(f"R-squared Score: {r2}")

    # Feature Importance
    feature_importance = grid_search.best_estimator_.named_steps['rf'].feature_importances_
    for i, importance in enumerate(feature_importance):
        logger.info(f"Feature {i+1} importance: {importance}")

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.savefig('actual_vs_predicted.png')
    logger.info("Saved actual vs predicted plot")

    # Compare with another model
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_scores = cross_val_score(gb_model, X, y, cv=5, scoring='neg_mean_squared_error')
    logger.info(f"Gradient Boosting CV MSE: {-gb_scores.mean()}")

    logger.info("Expanded Machine Learning task completed")


def perform_nhd_ml_task(file_path: str):
    logger.info("Starting NHD Flowline Machine Learning task")
    nhd_data = gpd.read_file(file_path)
    logger.debug(f"Loaded NHD data with {len(nhd_data)} flowlines")
    features = ['lengthkm', 'mainstemid']
    target = 'edhfcode'

    logger.info(f"Preparing features: {features} and target: {target}")
    X = nhd_data[features]
    y = nhd_data[target]

    logger.info("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

    logger.info("Setting up preprocessing pipeline")
    numeric_features = ['lengthkm']
    categorical_features = ['mainstemid']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    logger.info("Starting hyperparameter tuning")
    param_grid = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [None, 10, 20, 30]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')

    logger.info("Fitting the model (this may take a while)")
    grid_search.fit(X_train, y_train)

    logger.info(f"Best parameters: {grid_search.best_params_}")

    y_pred = grid_search.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    logger.info("***Model Performance:***")
    logger.info(f"Accuracy Score: {accuracy}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))

    # Feature Importance
    feature_importance = grid_search.best_estimator_.named_steps['rf'].feature_importances_
    feature_names = (numeric_features +
                     grid_search.best_estimator_.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_features).tolist())

    for name, importance in zip(feature_names, feature_importance):
        logger.info(f"Feature {name} importance: {importance}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix of EDHFCODE Predictions')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('edhfcode_confusion_matrix.png')
    plt.close()

    # Feature Importance Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=feature_importance, y=feature_names)
    plt.title('Feature Importance for EDHFCODE Prediction')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('edhfcode_feature_importance.png')
    plt.close()

    logger.info("Saved EDHFCODE prediction visualizations")
    logger.info("NHD Flowline Machine Learning task completed")


if __name__ == "__main__":
    logger.info("Starting all operations")

    perform_dask_operations()
    perform_graph_operations()
    combined_dask_and_dfs()
    perform_ml_task()
    expanded_ml_task()
    perform_nhd_ml_task("nhd_flowline.shp")

    logger.info("All operations completed successfully")
