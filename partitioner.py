import random
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.lines import Line2D


def get_incoming_edges(graph, node):
    """
    Get all incoming edges to the given node.

    :param graph: nx.DiGraph, the graph to analyze
    :param node: the target node to find incoming edges for

    :return: list of tuples representing incoming edges
    """
    incoming_edges = []
    # Loop over each node of the graph
    for u in graph:
        # Check if node is in the adjacency list of u
        if node in graph[u]:
            edge_data = graph[u][node]
            incoming_edges.append((u, node, edge_data))
    return incoming_edges


def get_outgoing_edges(graph, node):
    """
    Get all outgoing edges from the given node.

    :param graph: nx.DiGraph, the graph to analyze
    :param node: the source node to find outgoing edges for

    :return: list of tuples representing outgoing edges
    """
    outgoing_edges = []
    if node in graph:
        # Loop over each outgoing edge of the given node
        # Using .items() to avoid error caused by AtlasView
        for v, edge_data in graph[node].items():
            outgoing_edges.append((node, v, edge_data))
    return outgoing_edges


def get_edge_weight(graph, edge):
    """
    Calculate edge weight based on average passengers.

    :param graph: nx.DiGraph, the graph to analyze
    :param edge: an edge of the graph to compute the weight

    :return: the weight associated to the given edge
    """
    return float(graph[edge[0]][edge[1]].get('avg_passengers', 0))


class GraphPartitioner:
    def __init__(self, graph):
        self.graph = graph

    def merge_nodes(self, temp_graph, u, v, node_groups):
        """
        Merge node v into node u, maintaining edge directions and updating edge attributes.

        :param temp_graph: a temporary graph used for the merging process
        :param u: the node into which v will be merged
        :param v: the node to merge into u
        :param node_groups: dictionary that tracks the grouping of nodes into super-nodes

        :return: updated temporary graph and node_groups
        """

        # Update the dictionary of super nodes merging v group into u group
        # In this way the dictionary reflect the current state of the super-nodes
        node_groups[u].update(node_groups[v])

        # Get all the incoming and outgoing edges for the node v in the temp_graph
        # before modifying the graph
        v_incoming_edges = get_incoming_edges(temp_graph, v)
        v_outgoing_edges = get_outgoing_edges(temp_graph, v)

        # Handle incoming edges
        for pred, _, edge_data in v_incoming_edges:
            if pred != u:  # Avoid self-loops
                if not temp_graph.has_edge(pred, u):
                    temp_graph.add_edge(pred, u, **edge_data)
                else:
                    # Update edge attributes by summing existing values
                    for key in edge_data:
                        temp_graph[pred][u][key] = temp_graph[pred][u].get(key, 0) + edge_data.get(key, 0)

        # Handle outgoing edges
        for _, succ, edge_data in v_outgoing_edges:
            if succ != u:  # Avoid self-loops
                if not temp_graph.has_edge(u, succ):
                    temp_graph.add_edge(u, succ, **edge_data)
                else:
                    # Update edge attributes by summing existing values
                    for key in edge_data:
                        temp_graph[u][succ][key] = temp_graph[u][succ].get(key, 0) + edge_data.get(key, 0)

        # Remove merged node from the graph
        temp_graph.remove_node(v)

        # Remove the node v from the super-node dictionary
        node_groups.pop(v, None)

        return temp_graph, node_groups

    def single_contraction(self, graph):
        """
        Performs a single trial of the contraction algorithm, since is a randomized algorithm
        running it multiple times ensures we are more likely to obtain the optimal solution.

        :param graph: the graph to partition

        :return:
            - list of cut edges
            - set of nodes in first partition
            - set of nodes in second partition
        """

        # Create a copy of the graph for the merged result
        # In this way we do not modify our original graph, we can also repeat for multiple runs
        copy_graph = graph.copy()

        # Create a dictionary, each key is a node and the respective value is a set
        # where we can store the nodes that belong to super-nodes
        node_groups = {node: {node} for node in copy_graph}

        # Until we remain with just two super-nodes
        while len(copy_graph) > 2:
            edges = list(copy_graph.edges())
            if not edges:
                break

            # Calculate weights for remaining edges
            edge_weights = {edge: get_edge_weight(copy_graph, edge) for edge in edges}

            # Pick randomly an edge with probability given by the edge_weight
            # Edges with 0 avg_passengers are less likely to be contracted
            try:
                edge = random.choices(
                    list(edge_weights.keys()),
                    weights=list(edge_weights.values()),
                    k=1)[0]
            # When only 0 avg_passengers edges are left, just pick one randomly
            # Sum of weights must be greater than 0
            except ValueError:
                edge = random.choice(edges)

            u, v = edge
            # Merge the nodes of the picked edge
            copy_graph, node_groups = self.merge_nodes(copy_graph, u, v, node_groups)

        # Get partitions
        remaining_nodes = [node for node in copy_graph]

        # The first partition is the corresponding set of the first node
        # the nodes into the first super-node
        partition1 = node_groups[remaining_nodes[0]]

        # The second partition is the set of all the other remaining nodes
        partition2 = set().union(*[node_groups[n] for n in remaining_nodes[1:]])

        # Find cut edges
        cut_edges = [(u, v) for u, v in graph.edges()
                     if (u in partition1 and v in partition2) or
                     (u in partition2 and v in partition1)]

        return cut_edges, partition1, partition2

    def find_minimum_cut(self, num_trials=100):
        """
        Find the minimum cut in the graph using the contraction algorithm.

        :param self: the class object
        :param num_trials: number of trials to run

        :return:
            - list of cut edges
            - set of nodes in first partition
            - set of nodes in second partition, associated to the best solution
        """
        # Initialize the variables for the optimal solution
        best_cut_edges = None
        best_partition1 = None
        best_partition2 = None
        min_avg_passengers = float('inf')

        # Repeat the contraction algorithm for num_trials times
        for _ in range(num_trials):
            cut_edges, p1, p2 = self.single_contraction(self.graph)

            # We can count the number of average passengers in the cut by summing
            # the respective attribute stored in the edge data
            avg_passengers = sum(self.graph[u][v].get('avg_passengers', 0)
                                 for u, v in cut_edges)

            # If we obtain a better result than the previous one we update the best solution
            # so if we reduce the size of the cut, or we reduce the passenger affected from the cut
            if (best_cut_edges is None or avg_passengers < min_avg_passengers or
                (avg_passengers == min_avg_passengers and len(cut_edges) < len(best_cut_edges))):
                min_avg_passengers = avg_passengers
                best_cut_edges = cut_edges
                best_partition1 = p1
                best_partition2 = p2

        return best_cut_edges, best_partition1, best_partition2

    def visualize_network(self, title, partition1=None, partition2=None, cut_edges=None):
        """
        Visualize the flight network on US map with distinct partitions.

        :param title: title of the plot
        :param partition1: set of nodes in first partition
        :param partition2: set of nodes in second partition
        :param cut_edges: list of edges in the minimum cut

        :return: None
        """

        # Initialize the two subplots
        fig, ax = plt.subplots(figsize=(15, 10))

        # Load and plot US states map, using geopandas and the us-states.json file with the map
        us_states = gpd.read_file("us-states.json")
        us_states.plot(ax=ax, color='lightgray', edgecolor='white')

        # Create a dictionary where for each node we assign the relative coordinates(latitude and longitude)
        pos = {node: (self.graph.nodes[node]['longitude'],
                      self.graph.nodes[node]['latitude'])
               for node in self.graph}

        # Plot for the original network
        if partition1 is None and partition2 is None:
            # Draw all edges
            for edge in self.graph.edges():
                origin = pos[edge[0]]
                dest = pos[edge[1]]
                ax.plot([origin[0], dest[0]], [origin[1], dest[1]],
                        color='gray', linewidth=0.5, alpha=0.3)

            # Draw all nodes in pink, using networkx visualization for a better result
            nx.draw_networkx_nodes(self.graph, pos, node_color='hotpink',
                                   node_size=7, ax=ax)

        # Plot for the partitioned network
        else:
            # Create subgraphs for each partition
            G1 = self.graph.subgraph(partition1)
            G2 = self.graph.subgraph(partition2)

            # Draw edges for partition 1
            for edge in G1.edges():
                origin = pos[edge[0]]
                dest = pos[edge[1]]
                ax.plot([origin[0], dest[0]], [origin[1], dest[1]],
                        color='lightgray', linewidth=0.5, alpha=0.3)

            # Draw edges for partition 2
            for edge in G2.edges():
                origin = pos[edge[0]]
                dest = pos[edge[1]]
                ax.plot([origin[0], dest[0]], [origin[1], dest[1]],
                        color='lightgreen', linewidth=0.5, alpha=0.3)

            # Draw cut edges in red
            for edge in cut_edges:
                origin = pos[edge[0]]
                dest = pos[edge[1]]
                ax.plot([origin[0], dest[0]], [origin[1], dest[1]],
                        color='red', linewidth=3, alpha=1)

            # Draw nodes for each partition
            nx.draw_networkx_nodes(self.graph, pos, nodelist=list(partition1),
                                   node_color='hotpink', node_size=7, ax=ax)
            nx.draw_networkx_nodes(self.graph, pos, nodelist=list(partition2),
                                   node_color='yellow', node_size=7, ax=ax)

        # Set map boundaries, to "zoom" in the interested zone
        margin = 5
        bounds = [
            min(pos[node][0] for node in self.graph.nodes()) - margin,
            max(pos[node][0] for node in self.graph.nodes()) + margin,
            min(pos[node][1] for node in self.graph.nodes()) - margin,
            max(pos[node][1] for node in self.graph.nodes()) + margin
        ]
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])

        # Add legend, for partitioned visualization
        if partition1 and partition2:
            legend_elements = [
                Line2D([0], [0], color='gray', lw=1, alpha=0.3,
                       label='Partition 1 Flights'),
                Line2D([0], [0], color='green', lw=1, alpha=0.3,
                       label='Partition 2 Flights'),
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='hotpink', markersize=10,
                       label='Partition 1 Airports'),
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='yellow', markersize=10,
                       label='Partition 2 Airports'),
                Line2D([0], [0], color='red', lw=4, label='Cut Edges')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

        plt.title(title)
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

    def solve_and_visualize(self):
        """
        Solve the partition problem and visualize the results.

        :param self: the class object

        :return:
            - list of cut edges
            - set of nodes in first partition
            - set of nodes in second partition
        """
        # Visualize original network
        self.visualize_network("Original Flight Network")

        # Find minimum cut using contraction algorithm
        cut_edges, partition1, partition2 = self.find_minimum_cut()

        # Calculate and display average passengers in cut edges
        avg_passengers = sum(self.graph[u][v].get('avg_passengers', 0)
                             for u, v in cut_edges)

        # Visualize partitioned network
        self.visualize_network(
            f"Partitioned Flight Network\nAvg Passengers in Cut: {avg_passengers:.2f}",
            partition1, partition2, cut_edges
        )

        # Print the results
        print("\nResults:")
        print(f"Flights to remove: {cut_edges}")
        print(f"Average passengers across cut edges: {avg_passengers:.2f}")
        print(f"Airports partition 1 size: {len(partition1)}")
        print(f"Airports partition 2 size: {len(partition2)}")

        return cut_edges, partition1, partition2
