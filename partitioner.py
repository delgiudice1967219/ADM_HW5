import random
from typing import List, Tuple, Set, Dict, Optional
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.lines import Line2D


def _get_predecessors(graph: nx.DiGraph, node: str) -> Set[str]:
    """
    Get all nodes that have edges pointing to the given node.

    Args:
        graph: The graph to analyze
        node: The target node to find predecessors for

    Returns:
        Set of nodes that are predecessors of the given node
    """
    return {u for u, v in graph.edges() if v == node}


def _get_successors(graph: nx.DiGraph, node: str) -> Set[str]:
    """
    Get all nodes that the given node has edges pointing to.

    Args:
        graph: The graph to analyze
        node: The source node to find successors for

    Returns:
        Set of nodes that are successors of the given node
    """
    return {v for u, v in graph.edges() if u == node}


class GraphPartitioner:
    def __init__(self, graph: nx.DiGraph):
        """
        Initialize the GraphPartitioner with a directed graph.

        Args:
            graph (nx.DiGraph): Input graph with nodes and edges
        """
        self.graph = graph

    def _merge_vertices(self, graph: nx.DiGraph, u: str, v: str,
                        node_groups: Dict[str, Set[str]]) -> Tuple[nx.DiGraph, Dict[str, Set[str]]]:
        """
        Merges vertex v into u, maintaining edge directions.
        """
        # Create a new graph for the merged result
        H = graph.copy()
        # Update the set of super nodes merging v into u
        node_groups[u].update(node_groups[v])

        # Get predecessors and successors before modifying the graph
        v_predecessors = _get_predecessors(H, v)
        v_successors = _get_successors(H, v)

        # Handle incoming edges
        for pred in v_predecessors:
            if pred != u:  # Avoid self-loops
                edge_data = H[pred][v]
                if not H.has_edge(pred, u):
                    H.add_edge(pred, u, **edge_data)
                else:
                    # Update edge attributes by summing existing values
                    for key in edge_data:
                        H[pred][u][key] = H[pred][u].get(key, 0) + edge_data.get(key, 0)

        # Handle outgoing edges
        for succ in v_successors:
            if succ != u:  # Avoid self-loops
                edge_data = H[v][succ]
                if not H.has_edge(u, succ):
                    H.add_edge(u, succ, **edge_data)
                else:
                    # Update edge attributes by summing existing values
                    for key in edge_data:
                        H[u][succ][key] = H[u][succ].get(key, 0) + edge_data.get(key, 0)

        # Remove merged node
        H.remove_node(v)
        del node_groups[v]

        return H, node_groups

    def _get_edge_weight(self, graph: nx.DiGraph, edge: Tuple[str, str]) -> float:
        """
        Calculate edge weight based on average passengers.
        """
        return float(graph[edge[0]][edge[1]].get('avg_passengers', 0))

    def _single_trial(self, graph: nx.DiGraph) -> Tuple[List[Tuple[str, str]], Set[str], Set[str]]:
        """
        Performs a single trial of the contraction algorithm.
        """
        H = graph.copy()
        node_groups = {node: {node} for node in H}

        while len(H) > 2:
            edges = list(H.edges())
            if not edges:
                break

            # Calculate weights for remaining edges
            edge_weights = {edge: self._get_edge_weight(H, edge) for edge in edges}

            try:
                edge = random.choices(
                    list(edge_weights.keys()),
                    weights=list(edge_weights.values()),
                    k=1
                )[0]
            except ValueError:
                edge = random.choice(edges)

            u, v = edge
            H, node_groups = self._merge_vertices(H, u, v, node_groups)

        # Get partitions
        remaining_nodes = [node for node in H]
        if len(remaining_nodes) < 2:
            return [], set(), set()

        partition1 = node_groups[remaining_nodes[0]]
        partition2 = set().union(*[node_groups[n] for n in remaining_nodes[1:]])

        # Find cut edges
        cut_edges = [(u, v) for u, v in graph.edges()
                     if (u in partition1 and v in partition2) or
                     (u in partition2 and v in partition1)]

        return cut_edges, partition1, partition2

    def find_minimum_cut(self, num_trials: int = 100) -> Tuple[List[Tuple[str, str]], Set[str], Set[str]]:
        """
        Find the minimum cut in the graph using the contraction algorithm.

        Args:
            num_trials: Number of trials to run

        Returns:
            Tuple containing:
            - List of cut edges
            - Set of nodes in first partition
            - Set of nodes in second partition
        """
        best_cut_edges = None
        best_partition1 = None
        best_partition2 = None
        min_avg_passengers = float('inf')

        for _ in range(num_trials):
            cut_edges, p1, p2 = self._single_trial(self.graph)

            if not cut_edges:
                continue

            total_passengers = sum(self.graph[u][v].get('avg_passengers', 0)
                                   for u, v in cut_edges)
            avg_passengers = total_passengers / len(cut_edges)

            if avg_passengers < min_avg_passengers:
                min_avg_passengers = avg_passengers
                best_cut_edges = cut_edges
                best_partition1 = p1
                best_partition2 = p2

        if best_cut_edges is None:
            return [], set(), set()

        return best_cut_edges, best_partition1, best_partition2

    def visualize_network(self, title: str, partition1: Optional[Set[str]] = None,
                          partition2: Optional[Set[str]] = None,
                          cut_edges: Optional[List[Tuple[str, str]]] = None) -> None:
        """
        Visualize the flight network on US map with distinct partitions.
        """
        fig, ax = plt.subplots(figsize=(15, 10))

        # Load and plot US states boundaries
        us_states = gpd.read_file("us-states.json")
        us_states.plot(ax=ax, color='lightgray', edgecolor='white')

        pos = {node: (self.graph.nodes[node]['longitude'],
                      self.graph.nodes[node]['latitude'])
               for node in self.graph}

        if partition1 is None and partition2 is None:
            # Draw all edges in gray
            for edge in self.graph.edges():
                origin = pos[edge[0]]
                dest = pos[edge[1]]
                ax.plot([origin[0], dest[0]], [origin[1], dest[1]],
                        color='gray', linewidth=0.5, alpha=0.3)

            # Draw all nodes in pink
            nx.draw_networkx_nodes(self.graph, pos, node_color='hotpink',
                                   node_size=7, ax=ax)
        else:
            # Create subgraphs for each partition
            G1 = self.graph.subgraph(partition1)
            G2 = self.graph.subgraph(partition2)

            # Draw edges for partition 1 in blue
            for edge in G1.edges():
                origin = pos[edge[0]]
                dest = pos[edge[1]]
                ax.plot([origin[0], dest[0]], [origin[1], dest[1]],
                        color='lightgray', linewidth=0.5, alpha=0.3)

            # Draw edges for partition 2 in green
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

        # Set map boundaries
        margin = 5
        bounds = [
            min(pos[node][0] for node in self.graph.nodes()) - margin,
            max(pos[node][0] for node in self.graph.nodes()) + margin,
            min(pos[node][1] for node in self.graph.nodes()) - margin,
            max(pos[node][1] for node in self.graph.nodes()) + margin
        ]
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])

        # Add legend
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

    def solve_and_visualize(self) -> Tuple[List[Tuple[str, str]], Set[str], Set[str]]:
        """
        Solve the airline partition problem and visualize results on US map.

        Returns:
            Tuple containing:
            - List of cut edges
            - Set of nodes in first partition
            - Set of nodes in second partition
        """
        # Visualize original network
        self.visualize_network("Original Flight Network")

        # Find minimum cut using contraction algorithm
        cut_edges, partition1, partition2 = self.find_minimum_cut()

        # Calculate and display average passengers in cut edges
        avg_passengers = (sum(self.graph[u][v].get('avg_passengers', 0)
                              for u, v in cut_edges) / len(cut_edges))
        print(f"Average passengers across cut edges: {avg_passengers:.2f}")

        # Visualize partitioned network
        self.visualize_network(
            f"Partitioned Flight Network\nAvg Passengers in Cut: {avg_passengers:.2f}",
            partition1, partition2, cut_edges
        )

        return cut_edges, partition1, partition2
