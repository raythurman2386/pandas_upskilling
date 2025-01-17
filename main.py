import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


def read_csv_example():
    """Basic read CSV example"""
    data = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
    )
    print(data.head())
    print(f"Shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")


def group_by_example():
    """Group by example"""
    data = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
    )
    grouped = data.groupby("sex")
    print(grouped.size())
    print(grouped["total_bill"].mean())


def describe_example():
    """Describe example"""
    data = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
    )
    print(data.describe())


def correlation_example():
    """Correlation example"""
    data = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
    )
    print(data.corr())


def load_states_data() -> gpd.GeoDataFrame:
    """Helper function to load and prepare states data"""
    data = gpd.read_file("states_provinces.shp")
    data["area"] = data["geometry"].area
    return data


def states_basic_analysis():
    """Basic states visualization and analysis"""
    data = load_states_data()

    fig, ax = plt.subplots(figsize=(15, 10))
    data.plot(ax=ax, color="none", edgecolor="blue")
    ax.set_title("States and Provinces Overview")
    plt.show()


def states_area_analysis():
    """Analysis of state areas and distributions"""
    data = load_states_data()

    # Creates multiple visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

    # Area choropleth
    data.plot(column='area', ax=ax1, legend=True,
              legend_kwds={'label': 'Area'})
    ax1.set_title('State Areas')

    # Top 10 largest states
    top_10 = data.nlargest(10, 'area')
    top_10.plot(ax=ax2, color='red', alpha=0.5)
    ax2.set_title('10 Largest States/Provinces')

    # Area distribution histogram
    data['area'].hist(ax=ax3, bins=30)
    ax3.set_title('Distribution of State Areas')

    # Bar plot of top 10 areas
    top_10.plot(column='area', kind='bar', ax=ax4)
    ax4.set_title('Top 10 States by Area')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def states_spatial_analysis():
    """Spatial analysis of states"""
    data = load_states_data()
    # Calculate centroids and create new visualization
    data['centroid'] = data.geometry.centroid
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Plot states with centroids
    data.plot(ax=ax1, color='lightgrey', edgecolor='black')
    data.centroid.plot(ax=ax1, color='red', markersize=50, alpha=0.5)
    ax1.set_title('States with Centroids')

    # Calculate and plot state boundaries
    boundaries = data.boundary
    boundaries.plot(ax=ax2, color='blue')
    ax2.set_title('State Boundaries')

    plt.show()


def states_comprehensive():
    """Comprehensive analysis of states data"""
    data = load_states_data()
    data['perimeter'] = data.geometry.length
    data['compactness'] = 4 * np.pi * data['area'] / (data['perimeter'] ** 2)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    data.plot(column='compactness', ax=ax1, legend=True,
              legend_kwds={'label': 'Compactness Ratio'})
    ax1.set_title('State Compactness')

    # Area vs Perimeter scatter
    data.plot(kind='scatter', x='area', y='perimeter', ax=ax2)
    ax2.set_title('Area vs Perimeter')

    # Compactness distribution
    data['compactness'].hist(ax=ax3, bins=30)
    ax3.set_title('Distribution of Compactness Ratios')

    # Custom visualization
    data.plot(ax=ax4, column='area', cmap='viridis',
              legend=True, legend_kwds={'label': 'Area'})
    data.centroid.plot(ax=ax4, color='red', markersize=data['compactness']*100)
    ax4.set_title('States with Scaled Centroids')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    functions = {
        1: read_csv_example,
        2: group_by_example,
        3: describe_example,
        4: correlation_example,
        5: states_basic_analysis,
        6: states_area_analysis,
        7: states_spatial_analysis,
        8: states_comprehensive
    }

    while True:
        print("\nAvailable examples:")
        for num, func in functions.items():
            print(f"{num}. {func.__doc__}")
        print("0. Exit")

        choice = input("Enter the number of the example you want to run (0 to exit): ")
        if choice == "0":
            break
        elif choice.isdigit() and int(choice) in functions:
            functions[int(choice)]()
        else:
            print("Invalid choice. Please try again.")