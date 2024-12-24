import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px


def classify_graph_density(density):
    """
    Function that classify the graph based on its density.

    :param density: density of the graph to analyze

    :return: None
    """

    if density > 0.5:
        print("\nThe graph is dense.")
    elif density < 0.1:
        print("\nThe graph is sparse.")
    else:
        print("\nThe graph has moderate density.")


def graph_density(number_nodes, number_edges):
    """
    Compute the graph density.

    :param number_nodes: number of nodes of the graph to analyze
    :param number_edges: number of edges of the graph to analyze

    :return: the density of the graph (float)
    """

    density = number_edges / (number_nodes * (number_nodes - 1))
    return density


def identify_and_visualize_busiest_routes(flow_data_df):
    """
    Identify and plot all the busiest routes.

    :param flow_data_df: pandas DataFrame containing all busiest routes information

    :return: None
    """

    # Sort and identify the top 10 busiest routes by passenger flow
    busiest_routes = flow_data_df.sort_values(by='Passengers', ascending=False).head(20)

    # Print the busiest routes
    print("\nBusiest routes by passenger traffic:")
    print(busiest_routes)

    plt.figure(figsize=(10, 6))

    # Get the values for color mapping
    passenger_values = busiest_routes['Passengers'].values

    # Normalize the values for the color map (so the colors scale appropriately)
    norm = plt.Normalize(vmin=min(passenger_values), vmax=max(passenger_values))

    # Create the horizontal bar plot with color mapping
    plt.barh(busiest_routes['Origin_city'] + " to " + busiest_routes['Destination_city'],
             passenger_values, color=plt.cm.viridis(norm(passenger_values)))

    plt.xlabel('Passengers')
    plt.ylabel('Routes')
    plt.title('Top 20 Busiest Routes by Passenger Traffic')

    # Reverse order for top to bottom bars
    plt.gca().invert_yaxis()

    # Add a color bar to show the color scale
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=norm), ax=plt.gca())
    cbar.set_label('Number of Passengers')
    plt.show()


class FlightNetwork:
    def __init__(self, graph, df):
        self.graph = graph
        self.df = df

    def compute_number_nodes(self):
        """
        Count the number of nodes in the graph.

        :param self: the class object

        :return: number of nodes in the graph (int)
        """

        number_nodes = len(self.graph)
        return number_nodes

    def compute_number_edges(self):
        """
        Count the number of edges in the graph.

        :param self: the class object

        :return: number of edges in the graph (int)
        """

        number_edges = sum(len(self.graph[node]) for node in self.graph)
        return number_edges

    def compute_in_and_out_degrees(self):
        """
        Count the number of incoming and outcoming edges for each node.

        :param self: the class object

        :return: Two dictionaries in_degrees, out_degrees,
            containing the number of incoming and outcoming edges for each node
        """

        in_degrees = {node: 0 for node in self.graph}
        for origin in self.graph:
            for destination in self.graph[origin]:
                in_degrees[destination] += 1

        out_degrees = {}
        for node in self.graph:
            out_degrees[node] = len(self.graph[node])
        return in_degrees, out_degrees

    def plot_degree(self):
        """
        Plot the distribution of the number of incoming and outcoming edges for each node.

        :param self: the class object

        :return: None
        """

        in_degrees, out_degrees = self.compute_in_and_out_degrees()

        # Plot two separate histograms for the in-degree and the out-degree, respectively
        plt.hist(list(in_degrees.values()), bins=50, alpha=0.5, edgecolor='black')
        plt.xlabel("In-degree")
        plt.ylabel("Frequency")
        plt.show()

        plt.hist(list(out_degrees.values()), bins=50, alpha=0.5, edgecolor='black')
        plt.xlabel("Out-degree")
        plt.ylabel("Frequency")
        plt.show()

    def identify_hubs(self):
        """
        Identify all the hubs(airports with degrees higher than the 90th percentile) and represent as tabular data.

        :param self: the class object

        :return: pandas DataFrame with detailed information for each hub
        """

        # Compute the in-degree and out-degree for each node
        # Callback the respective function
        in_degrees, out_degrees = self.compute_in_and_out_degrees()

        # Combine in-degrees and out-degrees to get total degree for each node
        total_degrees = {node: in_degrees[node] + out_degrees[node] for node in self.graph}

        # Calculate the 90th percentile of total degrees
        degree_values = list(total_degrees.values())
        percentile_90 = np.percentile(degree_values, 90)

        # Identify hubs: nodes with total degree greater than the 90th percentile
        hubs = [node for node, degree in total_degrees.items() if degree > percentile_90]

        # Create a list to store the data
        hub_data = []

        # Extract all the information for each hub, store in a list of dictionary
        # in this way we can create a tabular representation of the hubs, using a pandas DataFrame
        for hub in hubs:
            node_data = {
                'Airport Code': hub,
                'City': self.graph.nodes[hub].get('city', 'N/A'),
                'Population': self.graph.nodes[hub].get('population', 'N/A'),
                'Total Degree': total_degrees.get(hub, 0)
            }
            hub_data.append(node_data)

        # Create a DataFrame
        hub_df = pd.DataFrame(hub_data)

        # Return the DataFrame just created, in this way it can be callable
        return hub_df

    def analyze_graph_features(self):
        """
        Function to compute the fundamental information about the graph.

        :param self: the class object:

        :return:
            - number_nodes (int): number of nodes in the graph
            - number_edges (int): number of edges in the graph
            - density (float): density of the graph.
        """

        number_nodes = self.compute_number_nodes()
        number_edges = self.compute_number_edges()
        density = graph_density(number_nodes, number_edges)
        return number_nodes, number_edges, density

    def summarize_graph_features(self, number_nodes, number_edges, density):
        """
        Function that generates a detailed report of the graph's features.

        :param number_nodes: number of nodes in the graph

        :param number_edges: number of edges in the graph

        :param density: density of the graph

        :return: None
        """

        print(f"\nNumber of nodes: {number_nodes}")

        print(f"\nNumber of edges: {number_edges}")

        print(f"\nDensity of the graph: {density}")

        classify_graph_density(density)

        print(f"\nHubs (Airports with degrees higher than the 90th percentile):")

        hubs_df = self.identify_hubs()

        hubs_df.sort_values(by='Total Degree', ascending=False, inplace=True)

        print(hubs_df.head(20))

        self.plot_degree()

    def compute_passenger_flow(self):
        """
        Function to compute the total passenger flow between each origin and destination city
        using the graph structure (nodes and edges) instead of the original DataFrame.

        :param self: the class object

        :return: pandas DataFrame containing the total passenger flow between each origin and destination city
        """

        # Create a list to store the flow data
        flow_data = []

        # Iterate over the edges of the graph to get the passenger flow
        for origin, destination, data in self.graph.edges(data=True):
            # Extract relevant data from the graph
            origin_city = self.graph.nodes[origin]['city']
            destination_city = self.graph.nodes[destination]['city']
            passengers = data['passengers']

            # Append the data to the flow_data list
            flow_data.append({
                'Origin_city': origin_city,
                'Destination_city': destination_city,
                'Passengers': passengers
            })

        # Convert the flow_data list to a DataFrame
        flow_data_df = pd.DataFrame(flow_data)
        return flow_data_df

    def calculate_avg_passengers_per_flight(self):
        """
        Function to calculate and plot the average passengers per flight.

        :param self: the class object

        :return: None
        """

        # Calculate the total passengers and flights for each route
        grouped_data = self.df.groupby(['Origin_airport', 'Destination_airport']).agg(
            {'Passengers': 'sum', 'Flights': 'sum'}).reset_index()

        # Calculate the average passengers per flight
        grouped_data['Average Passengers'] = grouped_data['Passengers'] / grouped_data['Flights']

        # Create a column for the route name
        grouped_data['Route'] = grouped_data['Origin_airport'] + " to " + grouped_data['Destination_airport']
        avg_passengers = grouped_data[['Route', 'Average Passengers']]

        print("Top 10 routes for passenger efficiency")
        print(avg_passengers.sort_values(by='Average Passengers', ascending=False).head(10))
        print("\nLast 10 routes for passenger efficiency")
        print(avg_passengers.sort_values(by='Average Passengers', ascending=False).tail(10))

        # Scatter plot
        fig = px.scatter(
            avg_passengers,
            x='Route',
            y='Average Passengers',
            size='Average Passengers',
            color='Average Passengers',
            title="Average Passengers per Flight for Each Route",
            labels={"Passengers": "Average Passengers per Flight", "Route": "Routes"},
            template="plotly_white"
        )

        # Add the opacity for the marker, in this way we improve the readability for the points that are overlapping
        fig.update_traces(marker=dict(opacity=0.6))
        fig.update_layout(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showgrid=True),
            height=600
        )
        fig.show()
