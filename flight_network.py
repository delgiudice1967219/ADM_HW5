import matplotlib.pyplot as plt
import numpy as np


class FlightNetwork:
    def __init__(self, graph, df):
        self.graph = graph
        self.df = df

    def compute_number_nodes(self):
        number_nodes = len(self.graph.nodes())
        return number_nodes

    def compute_number_edges(self):
        number_edges = len(self.graph.edges())
        return number_edges

    def graph_density(self, number_nodes, number_edges):
        density = (2 * number_edges) / number_nodes * (number_nodes - 1)
        return density

    def plot_degree(self):
        # For each node: Compute the in-degree and out-degree
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())

        # Plot two separate histograms for the in-degree and the out-degree, respectively
        plt.hist(list(in_degrees.values()), bins=20, alpha=0.5)
        plt.xlabel("In-degree")
        plt.ylabel("Frequency")
        plt.show()

        plt.hist(list(out_degrees.values()), bins=20, alpha=0.5)
        plt.xlabel("Out-degree")
        plt.ylabel("Frequency")
        plt.show()

    def identify_hubs(self):
        # Compute the in-degree and out-degree for each node
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())

        # Combine in-degrees and out-degrees to get total degree for each node
        total_degrees = {node: in_degrees[node] + out_degrees[node] for node in self.graph.nodes()}

        # Calculate the 90th percentile of total degrees
        degree_values = list(total_degrees.values())
        percentile_90 = np.percentile(degree_values, 90)

        # Identify hubs: nodes with total degree greater than the 90th percentile
        hubs = [node for node, degree in total_degrees.items() if degree > percentile_90]

        return hubs

    def classify_graph_density(self, density):
        if density > 0.5:
            print("The graph is dense.")
        elif density < 0.1:
            print("The graph is sparse.")
        else:
            print("The graph has moderate density.")

    def analyze_graph_features(self):
        number_nodes = self.compute_number_nodes()
        number_edges = self.compute_number_edges()
        density = self.graph_density(number_nodes, number_edges)
        self.classify_graph_density(density)
        return number_nodes, number_edges, density

    def summarize_graph_features(self, number_nodes, number_edges, density):
        print(f"Number of nodes: {number_nodes}")
        print(f"Number of edges: {number_edges}")
        print(f"Density of the graph: {density}")
        print(f"Hubs (Airports with degrees higher than the 90th percentile):")
        hubs = self.identify_hubs()
        for hub in hubs:
            print(f" - {hub}")
        self.plot_degree()

    def compute_passenger_flow(self):
        # Compute total passenger flow between each origin and destination city
        flow_data = self.df.groupby(['Origin_city', 'Destination_city'])['Passengers'].sum().reset_index()

        print("\nTotal passenger flow between origin and destination cities:")
        print(flow_data)

        # Store for later use in visualizing busiest routes
        self.flow_data = flow_data

    def identify_and_visualize_busiest_routes(self):
        # Sort and identify the busiest routes by passenger flow
        busiest_routes = self.flow_data.sort_values(by='Passengers', ascending=False).head(10)

        # Print the busiest routes
        print("\nBusiest routes by passenger traffic:")
        print(busiest_routes)

        # Visualize the busiest routes
        plt.figure(figsize=(10, 6))
        plt.barh(busiest_routes['Origin_city'] + " to " + busiest_routes['Destination_city'],
                 busiest_routes['Passengers'], color='green')
        plt.xlabel('Passengers')
        plt.ylabel('Routes')
        plt.title('Top 10 Busiest Routes by Passenger Traffic')
        plt.gca().invert_yaxis()  # Reverse order for top to bottom bars
        plt.show()

    def calculate_avg_passengers_per_flight(self):
        # Calculate the average passengers per flight for each route
        avg_passengers = self.df.groupby(['Origin_city', 'Destination_city'])['Passengers'].mean().reset_index()

        print("\nAverage passengers per flight for each route:")
        print(avg_passengers)

        # Visualize under/over-utilized routes
        plt.figure(figsize=(10, 6))
        plt.barh(avg_passengers['Origin_city'] + " to " + avg_passengers['Destination_city'],
                 avg_passengers['Passengers'], color='blue')
        plt.xlabel('Average Passengers per Flight')
        plt.ylabel('Routes')
        plt.title('Average Passengers per Flight for Each Route')
        plt.gca().invert_yaxis()
        plt.show()
