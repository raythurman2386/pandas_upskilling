import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, Polygon
import networkx as nx
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FakeGeospatialDataGenerator:
    def __init__(self, data_type="points", region="new_york", num_points=1000):
        self.data_type = data_type
        self.region = region
        self.num_points = num_points
        self.data = None
        self.boundary = None
        self.DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.graph = None
        self.street_network = None

        # Load or create boundary
        self.load_boundary()
        self.generate_street_network()

    def load_boundary(self):
        """Load boundary data for the specified region"""
        try:
            # You would typically load actual shapefiles here
            # For demonstration, creating simple boundaries
            boundaries = {
                "seattle": Polygon(
                    [
                        (-122.4359, 47.5003),
                        (-122.4359, 47.7340),
                        (-122.2359, 47.7340),
                        (-122.2359, 47.5003),
                        (-122.4359, 47.5003),
                    ]
                ),
                "new_york": Polygon(
                    [
                        (-74.2557, 40.4957),
                        (-74.2557, 40.9176),
                        (-73.7002, 40.9176),
                        (-73.7002, 40.4957),
                        (-74.2557, 40.4957),
                    ]
                ),
            }

            if self.region not in boundaries:
                raise ValueError(f"Region {self.region} not found")

            self.boundary = gpd.GeoDataFrame(
                {"geometry": [boundaries[self.region]]}, crs="EPSG:4326"
            )

            logger.info(f"Loaded boundary for {self.region}")

        except Exception as e:
            logger.error(f"Error loading boundary: {str(e)}")
            raise

    def generate_random_points(self):
        """Generate random points within boundary"""
        try:
            minx, miny, maxx, maxy = self.boundary.total_bounds

            # Generate more points than needed to account for points outside boundary
            factor = 1.5
            points = []
            while len(points) < self.num_points:
                # Generate random points
                x = np.random.uniform(minx, maxx, int(self.num_points * factor))
                y = np.random.uniform(miny, maxy, int(self.num_points * factor))

                # Create Points and check if they're within boundary
                for i in range(len(x)):
                    point = Point(x[i], y[i])
                    if self.boundary.geometry.iloc[0].contains(point):
                        points.append(point)
                        if len(points) >= self.num_points:
                            break

            return gpd.GeoDataFrame(geometry=points[: self.num_points], crs="EPSG:4326")

        except Exception as e:
            logger.error(f"Error generating points: {str(e)}")
            raise

    def generate_street_network(self):
        """Generate a fake street network using NetworkX"""
        try:
            # Get boundary bounds
            minx, miny, maxx, maxy = self.boundary.total_bounds

            # Create a grid-like street network
            n_grid = 20  # 20x20 grid
            G = nx.grid_2d_graph(n_grid, n_grid)

            # Convert grid coordinates to geo coordinates
            pos = {}
            node_gdf_data = []

            x_spacing = (maxx - minx) / (n_grid - 1)
            y_spacing = (maxy - miny) / (n_grid - 1)

            for i in range(n_grid):
                for j in range(n_grid):
                    # Convert grid coordinates to geographic coordinates
                    x = minx + i * x_spacing
                    y = miny + j * y_spacing

                    # Add small random offset
                    x += np.random.normal(0, x_spacing / 10)
                    y += np.random.normal(0, y_spacing / 10)

                    node = (i, j)
                    pos[node] = (x, y)
                    node_gdf_data.append(
                        {"node_id": f"{i}_{j}", "geometry": Point(x, y)}
                    )

            # Create edges (streets) with attributes
            edges_gdf_data = []
            for edge in G.edges():
                start_pos = pos[edge[0]]
                end_pos = pos[edge[1]]

                # Create street segment
                street = LineString([start_pos, end_pos])

                # Add street attributes
                edges_gdf_data.append(
                    {
                        "edge_id": f"{edge[0]}_{edge[1]}",
                        "geometry": street,
                        "length": street.length * 111000,  # Approximate meters
                        "speed_limit": np.random.choice([25, 30, 35, 40, 45]),  # mph
                        "street_type": np.random.choice(
                            ["residential", "arterial", "collector"]
                        ),
                    }
                )

            # Create GeoDataFrames for nodes and edges
            self.nodes_gdf = gpd.GeoDataFrame(node_gdf_data, crs="EPSG:4326")
            self.street_network = gpd.GeoDataFrame(edges_gdf_data, crs="EPSG:4326")

            # Create NetworkX graph with geographic coordinates
            self.graph = nx.Graph()

            # Add nodes with positions
            for node, position in pos.items():
                self.graph.add_node(node, pos=position)

            # Add edges with weights based on length
            for edge in G.edges():
                start_pos = pos[edge[0]]
                end_pos = pos[edge[1]]
                weight = (
                    (start_pos[0] - end_pos[0]) ** 2 + (start_pos[1] - end_pos[1]) ** 2
                ) ** 0.5
                self.graph.add_edge(edge[0], edge[1], weight=weight)

            logger.info("Successfully generated street network")

        except Exception as e:
            logger.error(f"Error generating street network: {str(e)}")
            raise

    def generate_routes(self):
        """Generate fake route data using the street network"""
        try:
            routes = []
            route_data = []

            # Get all nodes as a list of tuples
            nodes = list(self.graph.nodes())

            for i in range(self.num_points):
                # Select random start and end nodes
                start_node = nodes[np.random.randint(0, len(nodes))]
                end_node = nodes[np.random.randint(0, len(nodes))]

                # Ensure start and end nodes are different
                while start_node == end_node:
                    end_node = nodes[np.random.randint(0, len(nodes))]

                try:
                    # Find shortest path between nodes
                    path = nx.shortest_path(
                        self.graph, start_node, end_node, weight="weight"
                    )

                    # Convert path to coordinates
                    path_coords = []
                    for node in path:
                        pos = self.graph.nodes[node]["pos"]
                        path_coords.append((pos[0], pos[1]))

                    # Create route line
                    route = LineString(path_coords)

                    # Calculate route metrics
                    route_length = route.length * 111000  # Approximate meters
                    avg_speed = np.random.uniform(20, 40)  # mph
                    duration = (route_length / 1609.34) / avg_speed * 60  # minutes

                    routes.append(route)
                    route_data.append(
                        {
                            "route_id": i,
                            "distance": route_length,
                            "duration": duration,
                            "vehicle_type": np.random.choice(["car", "truck", "bike"]),
                            "start_time": datetime.now()
                            + timedelta(minutes=np.random.randint(0, 1440)),
                            "avg_speed": avg_speed,
                            "num_stops": len(path) - 1,
                        }
                    )

                except nx.NetworkXNoPath:
                    logger.warning(
                        f"No path found between nodes {start_node} and {end_node}"
                    )
                    continue

            if not routes:
                raise ValueError("No valid routes were generated")

            # Create GeoDataFrame with routes
            routes_gdf = gpd.GeoDataFrame(route_data, geometry=routes, crs="EPSG:4326")

            return routes_gdf

        except Exception as e:
            logger.error(f"Error generating routes: {str(e)}")
            raise

    def generate_polygons(self):
        """Generate fake polygon data (e.g., service areas, zones) without overlaps"""
        try:
            # Generate center points for polygons with minimum distance constraint
            points_gdf = self.generate_random_points()

            polygons = []
            polygon_data = []
            buffer_distance = 0.002

            for idx, point in enumerate(points_gdf.geometry):
                attempts = 0
                max_attempts = 5

                while attempts < max_attempts:
                    # Create random polygon around point
                    num_points = np.random.randint(6, 10)
                    angles = np.linspace(0, 360, num_points)
                    # Randomize distances but keep them within a reasonable range
                    base_distance = np.random.uniform(0.001, 0.003)
                    distances = base_distance + np.random.uniform(-0.0005, 0.0005, len(angles))

                    polygon_points = []
                    for angle, distance in zip(angles, distances):
                        dx = distance * np.cos(np.radians(angle))
                        dy = distance * np.sin(np.radians(angle))
                        polygon_points.append((point.x + dx, point.y + dy))

                    # Close the polygon
                    polygon_points.append(polygon_points[0])

                    # Create the polygon
                    polygon = Polygon(polygon_points)

                    # Check if polygon is valid and within boundary
                    if not polygon.is_valid:
                        attempts += 1
                        continue

                    # Buffer the polygon slightly to ensure no exact edges touch
                    buffered = polygon.buffer(0.0001)

                    # Check if the polygon is within boundary
                    if not self.boundary.geometry.iloc[0].contains(buffered):
                        attempts += 1
                        continue

                    # Check for overlaps with existing polygons
                    is_overlapping = False
                    if polygons:
                        # Create a temporary GeoDataFrame with existing polygons
                        existing_polygons = gpd.GeoDataFrame(geometry=polygons)
                        # Buffer existing polygons and check for intersection
                        buffered_existing = existing_polygons.geometry.buffer(buffer_distance)
                        if any(buffered_existing.intersects(buffered)):
                            is_overlapping = True

                    if not is_overlapping:
                        polygons.append(polygon)
                        polygon_data.append({
                            'zone_id': idx,
                            'area': polygon.area * (111000 ** 2),  # Approximate square meters
                            'category': np.random.choice(['residential', 'commercial', 'industrial']),
                            'population': np.random.randint(100, 10000),
                            'density': np.random.uniform(1000, 5000)  # people per square km
                        })
                        break

                    attempts += 1

                if attempts == max_attempts:
                    logger.warning(f"Could not create non-overlapping polygon for point {idx}")

            if not polygons:
                raise ValueError("No valid polygons were generated")

            # Create GeoDataFrame with polygons
            polygons_gdf = gpd.GeoDataFrame(
                polygon_data,
                geometry=polygons,
                crs="EPSG:4326"
            )

            # Add color column for visualization
            polygons_gdf['color'] = polygons_gdf['category'].map({
                'residential': 'lightblue',
                'commercial': 'lightgreen',
                'industrial': 'salmon'
            })

            logger.info(f"Successfully generated {len(polygons_gdf)} non-overlapping polygons")
            return polygons_gdf

        except Exception as e:
            logger.error(f"Error generating polygons: {str(e)}")
            raise

    def generate_data(self):
        """Generate fake geospatial data based on type"""
        try:
            if self.data_type == "points":
                points_gdf = self.generate_random_points()
                # Add additional point attributes
                points_gdf["point_id"] = range(len(points_gdf))
                points_gdf["timestamp"] = [
                    datetime.now() + timedelta(minutes=i)
                    for i in range(len(points_gdf))
                ]
                points_gdf["value"] = np.random.normal(50, 10, len(points_gdf))
                self.data = points_gdf

            elif self.data_type == "routes":
                self.data = self.generate_routes()

            elif self.data_type == "polygons":
                self.data = self.generate_polygons()

            else:
                raise ValueError(f"Unknown data type: {self.data_type}")

            logger.info(f"Successfully generated {len(self.data)} {self.data_type}")
            return True

        except Exception as e:
            logger.error(f"Error generating data: {str(e)}")
            return False

    def plot_data(self):
        """Plot the generated geospatial data with street network"""
        try:
            if self.data is None:
                raise ValueError("No data to plot. Run generate_data first.")

            fig, ax = plt.subplots(figsize=(15, 10))

            # Plot boundary
            self.boundary.plot(ax=ax, alpha=0.1, color='gray')

            # Plot street network if it exists
            if hasattr(self, 'street_network'):
                self.street_network.plot(ax=ax, color='lightgray', alpha=0.5, linewidth=0.5)

            # Different plotting logic based on data type
            if self.data_type == "routes":
                # Plot routes with different colors based on vehicle type
                for vtype in self.data['vehicle_type'].unique():
                    mask = self.data['vehicle_type'] == vtype
                    self.data[mask].plot(
                        ax=ax,
                        alpha=0.6,
                        label=vtype,
                        linewidth=2
                    )
                plt.legend()

            elif self.data_type == "polygons":
                # Plot polygons with category-based colors and black borders
                self.data.plot(
                    ax=ax,
                    column='category',
                    categorical=True,
                    legend=True,
                    alpha=0.5,
                    edgecolor='black',
                    linewidth=0.5
                )

            else:
                # Default plotting for points or other data types
                self.data.plot(
                    ax=ax,
                    alpha=0.6,
                    column='point_id' if 'point_id' in self.data.columns else None,
                    cmap='viridis'
                )

            plt.title(f'{self.data_type.title()} in {self.region.title()}')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')

            # Save the plot
            plt.savefig(os.path.join(self.DATA_DIR, f'geospatial_{self.region}_{self.data_type}_plot.png'))
            plt.close()

            logger.info(f"Plot saved as 'geospatial_{self.region}_{self.data_type}_plot.png'")
            return True

        except Exception as e:
            logger.error(f"Error plotting data: {str(e)}")
            return False

    def save_data(self, format="geojson"):
        """Save the generated data to file"""
        try:
            if self.data is None:
                raise ValueError("No data to save. Run generate_data first.")

            filename = (
                f"{self.data_type}_{self.region}_{datetime.now().strftime('%Y%m%d')}"
            )
            os.makedirs(self.DATA_DIR, exist_ok=True)

            if format == "geojson":
                filepath = os.path.join(self.DATA_DIR, f"{filename}.geojson")
                self.data.to_file(filepath, driver="GeoJSON")
            elif format == "shapefile":
                filepath = os.path.join(self.DATA_DIR, f"{filename}.shp")
                self.data.to_file(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Data saved to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return None


def main():
    # Generate point data
    region = "seattle"
    point_generator = FakeGeospatialDataGenerator(
        data_type="points", region=region, num_points=1000
    )
    point_generator.generate_data()
    point_generator.plot_data()
    point_generator.save_data(format="geojson")

    # Generate route data
    route_generator = FakeGeospatialDataGenerator(
        data_type="routes", region=region, num_points=100
    )
    route_generator.generate_data()
    route_generator.plot_data()
    route_generator.save_data(format="geojson")

    # Generate polygon data
    polygon_generator = FakeGeospatialDataGenerator(
        data_type="polygons", region=region, num_points=50
    )
    polygon_generator.generate_data()
    polygon_generator.plot_data()
    polygon_generator.save_data(format="shapefile")


if __name__ == "__main__":
    main()
