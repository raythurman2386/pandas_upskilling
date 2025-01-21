import pandas as pd
import geopandas as gpd
import folium
from folium import plugins


class VolcanoAnalysis:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.gdf = self.load_volcano_data()

    def load_volcano_data(self) -> gpd.GeoDataFrame:
        """Helper function to load and prepare volcano data"""
        dataset = pd.read_csv(self.csv_file)
        data = dataset.loc[:, ("Year", "Name", "Country", "Latitude", "Longitude", "Type")]
        geometry = gpd.points_from_xy(data.Longitude, data.Latitude)
        return gpd.GeoDataFrame(data, crs="EPSG:4326", geometry=geometry)

    @staticmethod
    def get_volcano_type_color(volcano_type: str) -> str:
        """Helper function to determine marker color based on volcano type"""
        color_map = {
            "Stratovolcano": "green",
            "Caldera": "blue",
            "Complex volcano": "purple",
            "Lava dome": "orange"
        }
        return color_map.get(volcano_type, "red")

    def clean_geometry(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Clean geometry by keeping only the main geometry column"""
        geometry_columns = gdf.select_dtypes(include=['geometry']).columns
        if len(geometry_columns) > 1:
            for col in geometry_columns[1:]:
                gdf[col] = gdf[col].apply(lambda geom: geom.wkt if geom else None)
        return gdf

    def spatial_analysis(self):
        """Spatial analysis of volcano distributions"""
        self.gdf['buffer_zones'] = self.gdf.geometry.buffer(1)
        nearby_volcanoes = gpd.sjoin(self.gdf, self.gdf.geometry.buffer(2).to_frame('geometry'),
                                     how='left', predicate='within')
        self.gdf['nearest_distances'] = self.gdf.geometry.apply(lambda g:
                                                                self.gdf[self.gdf.geometry != g].distance(g).min())

        # Clean geometry before saving
        cleaned_gdf = self.clean_geometry(self.gdf)
        cleaned_gdf.to_file("volcano_spatial_analysis.gpkg", driver="GPKG")
        print("Spatial analysis completed. Results saved to volcano_spatial_analysis.gpkg")

    def density_analysis(self):
        """Density analysis of volcanoes"""
        volcano_counts = self.gdf.groupby('Country').size().sort_values(ascending=False)

        # Create a GeoDataFrame with the counts
        result = gpd.GeoDataFrame({
            'Country': volcano_counts.index,
            'Count': volcano_counts.values
        })

        # Merge with the original GeoDataFrame to get geometry
        result = result.merge(self.gdf[['Country', 'geometry']], on='Country', how='left')

        # Drop duplicates and set the geometry
        result = result.drop_duplicates(subset='Country').set_geometry('geometry')

        # Clean geometry before saving
        result = self.clean_geometry(result)

        result.to_file("volcano_density_analysis.gpkg", driver="GPKG")
        print("Density analysis completed. Results saved to volcano_density_analysis.gpkg")

    def temporal_analysis(self):
        """Temporal analysis of volcanic activity"""
        temporal_analysis = self.gdf.groupby('Year').size().reset_index(name='YearlyCount')
        self.gdf['Decade'] = (self.gdf['Year'] // 10) * 10
        decade_analysis = self.gdf.groupby('Decade').size().reset_index(name='DecadeCount')

        # Merge the yearly and decade analysis
        result = temporal_analysis.merge(decade_analysis, left_on='Year', right_on='Decade', how='outer')
        result['Decade'] = result['Decade'].fillna(result['Year'] // 10 * 10)
        result = result.sort_values('Year')

        # Create a point geometry for each year (this is just for demonstration, you might want to adjust this)
        result['geometry'] = gpd.points_from_xy(result['Year'], result['YearlyCount'])

        # Convert to GeoDataFrame
        result = gpd.GeoDataFrame(result, geometry='geometry', crs="EPSG:4326")

        # Clean geometry before saving
        result = self.clean_geometry(result)

        result.to_file("volcano_temporal_analysis.gpkg", driver="GPKG")
        print("Temporal analysis completed. Results saved to volcano_temporal_analysis.gpkg")

    def type_analysis(self):
        """Analysis of volcano types"""
        type_distribution = self.gdf.groupby('Type').size()
        result = gpd.GeoDataFrame({
            'Type': type_distribution.index,
            'Count': type_distribution.values
        })
        result.to_file("volcano_type_analysis.gpkg", driver="GPKG")
        print("Type analysis completed. Results saved to volcano_type_analysis.gpkg")

    def create_interactive_map(self):
        """Create interactive map with Folium"""
        volcano_map = folium.Map(location=[0, 0], tiles="OpenStreetMap", zoom_start=3)

        # Add original volcano data
        for idx, row in self.gdf.iterrows():
            point = [row.geometry.y, row.geometry.x]
            type_color = self.get_volcano_type_color(row.Type)

            folium.Marker(
                location=point,
                popup=f"Year: {row.Year} Name: {row.Name} Country: {row.Country} Type: {row.Type}",
                icon=folium.Icon(color=type_color),
            ).add_to(volcano_map)

        # Add spatial analysis data
        spatial_gdf = gpd.read_file("volcano_spatial_analysis.gpkg")
        folium.GeoJson(
            spatial_gdf,
            name='Spatial Analysis',
            style_function=lambda feature: {
                'fillColor': 'yellow',
                'color': 'black',
                'weight': 2,
                'fillOpacity': 0.5,
            }
        ).add_to(volcano_map)

        # Add density analysis data
        density_gdf = gpd.read_file("volcano_density_analysis.gpkg")
        folium.Choropleth(
            geo_data=density_gdf.to_json(),
            name='Density Analysis',
            data=density_gdf,
            columns=['Country', 'Count'],
            key_on='feature.properties.Country',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Volcano Count'
        ).add_to(volcano_map)

        # Add temporal analysis data
        temporal_gdf = gpd.read_file("volcano_temporal_analysis.gpkg")
        temporal_gdf['Year'] = temporal_gdf['Year'].astype(int)

        # Create a custom JSON-friendly dictionary
        styledict = {str(year): {
            'color': 'red',
            'opacity': 0.8,
            'fillColor': 'red',
            'fillOpacity': 0.5
        } for year in temporal_gdf['Year'].unique()}

        time_slider = plugins.TimeSliderChoropleth(
            temporal_gdf.to_json(),
            styledict=styledict
        ).add_to(volcano_map)

        # Add layer control
        folium.LayerControl().add_to(volcano_map)

        volcano_map.save("interactive_volcano_map.html")
        print("Interactive map created. Saved as interactive_volcano_map.html")


if __name__ == "__main__":
    volcano_analysis = VolcanoAnalysis("volcano_data.csv")

    analysis_functions = {
        1: volcano_analysis.spatial_analysis,
        2: volcano_analysis.density_analysis,
        3: volcano_analysis.temporal_analysis,
        4: volcano_analysis.type_analysis,
        5: volcano_analysis.create_interactive_map
    }

    while True:
        print("\nAvailable analyses:")
        for num, func in analysis_functions.items():
            print(f"{num}. {func.__doc__}")
        print("0. Exit")

        choice = input("Enter the number of the analysis you want to run (0 to exit): ")
        if choice == "0":
            break
        elif choice.isdigit() and int(choice) in analysis_functions:
            analysis_functions[int(choice)]()
        else:
            print("Invalid choice. Please try again.")