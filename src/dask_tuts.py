import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from datetime import datetime, timedelta
import random
import string

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


if __name__ == "__main__":
    logger.info("Starting combined Dask and Graph operations")

    perform_dask_operations()
    perform_graph_operations()

    logger.info("All operations completed successfully")