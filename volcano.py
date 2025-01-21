import pandas as pd
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
from shapely.ops import nearest_points, triangulate


def load_volcano_data() -> gpd.GeoDataFrame:
    """Helper function to load and prepare volcano data"""
    dataset = pd.read_csv("volcano_data.csv")
    data = dataset.loc[:, ("Year", "Name", "Country", "Latitude", "Longitude", "Type")]
    geometry = gpd.points_from_xy(data.Longitude, data.Latitude)
    return gpd.GeoDataFrame(data, crs="EPSG:4326", geometry=geometry)


def get_volcano_type_color(volcano_type: str) -> str:
    """Helper function to determine marker color based on volcano type"""
    color_map = {
        "Stratovolcano": "green",
        "Caldera": "blue",
        "Complex volcano": "purple",
        "Lava dome": "orange"
    }
    return color_map.get(volcano_type, "red")


def volcano_spatial_analysis():
    """Spatial analysis of volcano distributions"""
    gdf = load_volcano_data()

    # Create buffer zones and find clusters
    buffer_zones = gdf.buffer(1)
    nearby_volcanoes = gpd.sjoin(gdf, gdf.buffer(2).to_frame('geometry'),
                                 how='left', predicate='within')

    # Calculate nearest neighbor distances
    nearest_distances = gdf.geometry.apply(lambda g:
                                           gdf[gdf.geometry != g].distance(g).min())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    gdf.plot(ax=ax1, color='red', markersize=50, alpha=0.5)
    gpd.GeoSeries(buffer_zones).plot(ax=ax1, alpha=0.2)
    ax1.set_title('Volcano Buffer Zones')
    nearest_distances.hist(ax=ax2, bins=30)
    ax2.set_title('Distribution of Nearest Neighbor Distances')

    plt.show()


def volcano_density():
    """Density analysis of volcanoes"""
    gdf = load_volcano_data()
    fig, ax = plt.subplots(figsize=(15, 10))

    # Count volcanoes by country
    volcano_counts = gdf.groupby('Country').size().sort_values(ascending=False)
    volcano_counts.head(10).plot(kind='bar', ax=ax)
    plt.title('Top 10 Countries by Number of Volcanoes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def volcano_temporal():
    """Temporal analysis of volcanic activity"""
    gdf = load_volcano_data()
    temporal_analysis = gdf.groupby('Year').size()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    temporal_analysis.plot(ax=ax1)
    ax1.set_title('Volcanic Activity Over Time')

    # Decade-wise analysis
    gdf['Decade'] = (gdf['Year'] // 10) * 10
    decade_analysis = gdf.groupby('Decade').size()
    decade_analysis.plot(kind='bar', ax=ax2)
    ax2.set_title('Volcanic Activity by Decade')

    plt.tight_layout()
    plt.show()


def volcano_type_analysis():
    """Analysis of volcano types"""
    gdf = load_volcano_data()
    type_distribution = gdf.groupby('Type').size()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    type_distribution.plot(kind='pie', ax=ax1, autopct='%1.1f%%')
    ax1.set_title('Distribution of Volcano Types')

    for v_type in gdf['Type'].unique():
        subset = gdf[gdf['Type'] == v_type]
        color = get_volcano_type_color(v_type)
        subset.plot(ax=ax2, color=color, label=v_type, markersize=50, alpha=0.5)

    ax2.legend()
    ax2.set_title('Spatial Distribution by Volcano Type')

    plt.tight_layout()
    plt.show()


def volcano_interactive_map():
    """Create interactive map with Folium"""
    gdf = load_volcano_data()
    volcano_map = folium.Map(location=[0, 0], tiles="OpenStreetMap", zoom_start=4)
    for idx, row in gdf.iterrows():
        point = [row.geometry.y, row.geometry.x]
        type_color = get_volcano_type_color(row.Type)

        folium.Marker(
            location=point,
            popup=f"Year: {row.Year} Name: {row.Name} Country: {row.Country} Type: {row.Type}",
            icon=folium.Icon(color=type_color),
        ).add_to(volcano_map)

    volcano_map.save("volcano_map.html")


if __name__ == "__main__":
    functions = {
        1: volcano_spatial_analysis,
        2: volcano_density,
        3: volcano_temporal,
        4: volcano_type_analysis,
        5: volcano_interactive_map
    }

    while True:
        print("\nAvailable analyses:")
        for num, func in functions.items():
            print(f"{num}. {func.__doc__}")
        print("0. Exit")

        choice = input("Enter the number of the analysis you want to run (0 to exit): ")
        if choice == "0":
            break
        elif choice.isdigit() and int(choice) in functions:
            functions[int(choice)]()
        else:
            print("Invalid choice. Please try again.")