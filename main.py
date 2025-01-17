import pandas as pd
import geopandas as gpd
import folium
import geodatasets
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


def states_example():
    """Geopandas example"""
    data = gpd.read_file("states_provinces.shp")
    data["area"] = data["geometry"].area
    print(data.head(5))
    fig, ax = plt.subplots(figsize=(15, 10))
    data.plot(ax=ax, color="none", edgecolor="blue")
    ax.set_title("Antarctic ice shelves")
    plt.show()


def volcano_example():
    """Folium volcano example"""
    dataset = pd.read_csv("volcano_data.csv")
    data = dataset.loc[:, ("Year", "Name", "Country", "Latitude", "Longitude", "Type")]
    geometry = gpd.points_from_xy(data.Longitude, data.Latitude)
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326", geometry=geometry)
    volcano_map = folium.Map(location=[0, 0], tiles="OpenStreetMap", zoom_start=4)
    gdf_list = [[point.xy[1][0], point.xy[0][0]] for point in gdf.geometry]
    i = 0
    for point in gdf_list:
        if gdf.Type[i] == "Stratovolcano":
            type_color = "green"
        elif gdf.Type[i] == "Caldera":
            type_color = "blue"
        elif gdf.Type[i] == "Complex volcano":
            type_color = "yellow"
        elif gdf.Type[i] == "Lava dome":
            type_color = "orange"
        else:
            type_color = "red"
        folium.Marker(
            location=point,
            popup=f"Year: {str(gdf.Year[i])} Name: {gdf.Name[i]} Country: {gdf.Country[i]} Coordinates: {point} Type: {gdf.Type[i]}",
            radius=5,
            color=type_color,
            icon=folium.Icon(color="%s" % type_color),
            fill_color=type_color,
        ).add_to(volcano_map)
        i += 1

    volcano_map.save("map.html")


if __name__ == "__main__":
    functions = {
        1: read_csv_example,
        2: group_by_example,
        3: describe_example,
        4: correlation_example,
        5: states_example,
        6: volcano_example,
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
