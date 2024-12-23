import folium
from folium.plugins import MarkerCluster
import numpy as np


class FlightNetworkVisualizer:
    def __init__(self, graph, df, hubs_df):
        self.graph = graph
        self.df = df
        self.hubs_df = hubs_df

    def calculate_avg_passengers(self):
        """
        Computes average passengers per flight for each route
        :param self: the class object
        :return: pandas dataframe with average passengers per flight
        """
        # Create a DataFrame to store the average passengers for each route
        # We need that to filter the route based on this information

        # Calculate the total passengers and flights for each route
        grouped_data = self.df.groupby(['Origin_airport', 'Destination_airport']).agg(
            {'Passengers': 'sum', 'Flights': 'sum'}).reset_index()

        # Calculate the average passengers per flight
        grouped_data['Average Passengers'] = grouped_data['Passengers'] / grouped_data['Flights']

        avg_passengers_df = self.df.groupby(['Origin_airport', 'Destination_airport'])['Passengers'].mean().reset_index()
        return grouped_data

    def add_avg_passengers_to_edges(self, avg_passengers_df):
        """
        Add the computed average passengers to the edges of the graph
        :param self: the class object
        :param avg_passengers_df: the dataframe containing the average passengers per flight for each route
        :return: None
        """
        # Loop through each route and add average passengers to the edge
        for _, row in avg_passengers_df.iterrows():
            origin_airport = row['Origin_airport']
            destination_airport = row['Destination_airport']
            avg_passengers = row['Average Passengers']

            origin_node = origin_airport  # origin_airport is directly the node in the graph
            destination_node = destination_airport  # destination_airport is directly the node in the graph

            # If the graph has the relative edge add as an attribute the average passengers info
            if self.graph.has_edge(origin_node, destination_node):
                self.graph[origin_node][destination_node]['avg_passengers'] = avg_passengers

    def create_map(self, passenger_range, node_filter):
        """
        Create a folium interactive Map
        :param passenger_range: list of value of the slider filter inserted in input in the dash web app
        :param node_filter: the label relative to the filter inserted in input in the dash web app
        :return: HTML of the Folium Map
        """
        # Create a folium map centered in the United States, specifying the CartoDB dark_matter tile (dark stile)
        map = folium.Map(location=[37.0902, -95.7129], zoom_start=4, tiles='CartoDB dark_matter')

        # Add the marker cluster, crucial when there a lot of marker(nodes) on the map
        # It creates a single marker that collect all the neighbor when you zoom in it expand
        # This let us improve performance reducing the number of visualized markers,
        # that make faster interacting with the map
        marker_cluster = MarkerCluster().add_to(map)

        # Filter nodes based on user choice (hubs, non-hubs, or all)
        if node_filter == "hubs":
            nodes_to_display = [node for node in self.graph.nodes if node in self.hubs_df['Airport Code'].tolist()]
        elif node_filter == "non_hubs":
            nodes_to_display = [node for node in self.graph.nodes if node not in self.hubs_df['Airport Code'].tolist()]
        else:
            nodes_to_display = list(self.graph.nodes)

        # Add routes (edges) on the map
        for origin, destination, data in self.graph.edges(data=True):
            # Collect the info about the average passengers for each route, now stored into the edge
            avg_passengers = data.get('avg_passengers', 0)

            # Filter: if the passengers mean of the edge is in the range
            if passenger_range[0] <= avg_passengers <= passenger_range[1]:
                origin_node = self.graph.nodes[origin]
                destination_node = self.graph.nodes[destination]

                # Collect the coordinates for the origin and destination airport
                # Stored into each node
                origin_lat = origin_node['latitude']
                origin_lon = origin_node['longitude']
                destination_lat = destination_node['latitude']
                destination_lon = destination_node['longitude']

                # Do not consider the routes with NaN latitude and longitude
                # if we do not this, the map will not visualize raising an Exception(What type of exception?)
                if not (np.isnan(origin_lat) or np.isnan(origin_lon) or np.isnan(destination_lat) or np.isnan(
                        destination_lon)):
                    folium.PolyLine(locations=[(origin_lat, origin_lon), (destination_lat, destination_lon)],
                                    color='blue', weight=0.7, opacity=0.3).add_to(marker_cluster)

        # Add airport markers
        for node in nodes_to_display:
            node_data = self.graph.nodes[node]
            lat, lon = node_data['latitude'], node_data['longitude']

            # Verify coordinates are valid(not NaN)
            if np.isnan(lat) or np.isnan(lon):
                continue

            # Collect the information to insert into the popup
            city = node_data['city']
            population = node_data['population']

            # Representation for HUBs
            if node in self.hubs_df['Airport Code'].tolist():
                folium.Marker(location=[lat, lon],
                              # Red cloud icon
                              icon=folium.Icon(color='red', icon='cloud'),
                              # When you click on it, you can see its information
                              popup=folium.Popup(f"<b>HUB: {node}</b><br>{city}<br>Population: {population:.2f}",
                                                 max_width=500)).add_to(marker_cluster)
            else:
                # Representation for Not-Hub
                folium.Marker(location=[lat, lon],
                              # Green info-sign icon
                              icon=folium.Icon(color='green', icon='info-sign'),
                              # When you click on it, you can see its information
                              popup=folium.Popup(f"<b>{node}</b><br>{city}<br>Population: {population:.2f}",
                                                 max_width=500)).add_to(marker_cluster)

        # Return the Folium map converted in an HTML object
        return map._repr_html_()
