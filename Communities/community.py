import networkx as nx
from collections import defaultdict

class Community:
    def __init__(self, G, weight='distance') -> None:
        self.weight = weight
        self.result_map = {}

        # Sum of all edge weights in the graph
        self.m = G.size(weight=self.weight)

        self.graph = G

        # Maps and degrees for communities and nodes
        self.community_map = {}
        self.degree = {}  # Total degree of each node
        self.int_degree = {}  # Internal degree (self-loops) of each node
        self.community_degree = {}  # Total degree of each community
        self.community_int_degree = {}  # Total internal degree of each community

    def initialize(self):
        """Initialize communities with one node per community."""
        for comm, node in enumerate(self.graph.nodes()):
            # Map each node to its own community
            self.community_map[node] = comm

            # Degree of the node
            self.degree[node] = self.graph.degree(node, weight=self.weight)

            # Internal degree (self-loops)
            int_deg = 0
            if self.graph.has_edge(node, node):
                int_deg = self.graph[node][node].get(self.weight, 1)
            self.int_degree[node] = int_deg

            # Initialize community degrees
            self.community_degree[comm] = self.degree[node]
            self.community_int_degree[comm] = int_deg

    def get_community(self, node):
        """Return the community of a given node."""
        return self.community_map[node]

    def remove(self, node, comm, incident_weight):
        """Remove a node from a community and update statistics."""
        self.community_degree[comm] -= self.degree[node]
        self.community_int_degree[comm] -= self.int_degree[node] + incident_weight
        self.community_map[node] = None  # Node is no longer in any community

    def add(self, node, comm, incident_weight):
        """Add a node to a community and update statistics."""
        self.community_degree[comm] += self.degree[node]
        self.community_int_degree[comm] += self.int_degree[node] + incident_weight
        self.community_map[node] = comm

    def assign_best_community(self, node):
        """Find the best community for the given node."""
        incident_weight = defaultdict(int)
        
        # Calculate weights to all neighboring communities
        for neighbour in self.graph[node]:
            neighbour_comm = self.community_map[neighbour]
            incident_weight[neighbour_comm] += self.graph[node][neighbour].get(self.weight, 1)
        
        old_comm = self.community_map[node]
        old_weight = incident_weight[old_comm]
        self.remove(node, old_comm, old_weight)

        # Find the community with the maximum modularity gain
        best_comm = None
        best_mod = -float('inf')
        for comm, weight in incident_weight.items():
            mod = self.delta_mod(self.graph, node, comm, weight)
            if mod > best_mod:
                best_mod = mod
                best_comm = comm
        if best_comm is None:
            best_comm = old_comm
        self.add(node, best_comm, incident_weight[best_comm])
        
        if old_comm != best_comm:
            return True
        return False

    def delta_mod(self, G, node, community, incident_weight):
        """Calculate the change in modularity for adding a node to a community."""
        # Sum of the weights of the links incident to nodes in C
        sigma_tot = self.community_degree[community]

        # Sum of the weights of the links incident to node i
        k_i = self.degree[node]

        # Sum of the weights of the links from i to nodes in C
        k_i_in = incident_weight

        # Sum of the weights of all the links in the network
        m = self.m

        # Change in modularity
        delta_Q = (2 * k_i_in - sigma_tot * k_i / m) / (2 * m)
        return delta_Q

    def agglomerate(self):
        """Agglomerate the graph so that each node represents a community from the previous iteration."""
        # Create a new graph where nodes represent communities
        new_graph = nx.Graph()

        # Create mapping for community-to-community edges
        for node in self.graph.nodes():
            node_comm = self.community_map[node]  # Get the community of the node

            # Iterate over the neighbors of the node
            for neighbor in self.graph[node]:
                neighbor_comm = self.community_map[neighbor]  # Get the community of the neighbor
                
                # Add an edge between the communities, updating the weight
                weight = self.graph[node][neighbor].get(self.weight, 1)  # Get the edge weight, default to 1
                
                # Add the edge to the new graph (NetworkX will handle self-loops and weight updates)
                new_graph.add_edge(node_comm, neighbor_comm, weight=weight)

        # Replace the graph with the new agglomerated graph
        self.graph = new_graph

    def update_result_map(self):
        """Update the result_map so it reflects the mapping from the initial nodes to the final communities."""
        # If result_map is empty (first iteration), initialize it with community_map
        if not self.result_map:
            self.result_map = self.community_map.copy()
            return

        # Update the result_map based on the current community assignments
        for node, old_comm in self.result_map.items():
            new_comm = self.community_map.get(old_comm, old_comm)
            self.result_map[node] = new_comm

    def __iter__(self):
        """Iterate over the nodes in the community (graph)."""
        return iter(self.graph.nodes())

