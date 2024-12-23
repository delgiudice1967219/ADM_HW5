import heapq
import pandas as pd
#------------------------------------------------------------------------------------#   
class GraphMetrics:
    """ 
The GraphMetrics class provides various methods to analyze centrality measures in a directed graph using NetworkX. 
It includes functions to compute betweenness centrality, closeness centrality, degree centrality, PageRank, and Katz centrality. 
The class uses Dijkstra's algorithm to calculate shortest paths for centrality calculations. 
It also provides a method, analyze_centrality, that returns a DataFrame summarizing these centrality scores for a given node (airport).
    """
    def __init__(self, graph):
        """
        Initialize the network analyzer with a graph representation.

        Args:
            graph --> nx.DiGraph(): Directed Graph from Networkx Library
                     
        """
        self.graph = graph
        self.nodes = graph.nodes

#------------------------------------------------------------------------------------#   

    def shortest_paths(self, start:str):
        """
        Compute shortest paths from a start node using Dijkstra's algorithm.

        Args:
            start: Starting node

        Returns:
               A dictionary of shortest distances and paths
        """
        # Initialize distances and paths
        distances = {node: float('inf') for node in self.nodes}
        distances[start] = 0
        previous_nodes = {node: [] for node in self.nodes}
        unvisited = [(0, start)]   # List of tuples (distance, node)

        while unvisited:
            current_distance, current_node = heapq.heappop(unvisited) 
            # If a node has already been processed, skip it
            if current_distance > distances[current_node]:
                continue

             # Check all neighbors
            for neighbor, data in self.graph[current_node].items():
                weight = data.get('weight', 1)
                distance = current_distance + weight

                # If a shorter path is found
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = [current_node]
                    heapq.heappush(unvisited, (distance, neighbor)) 

                 # If an equally short path is found
                elif distance == distances[neighbor]:
                    previous_nodes[neighbor].append(current_node)

        return distances, previous_nodes
        
#------------------------------------------------------------------------------------#  
 
    #-------Define Centrality Measures-------#

    def betweenness_centrality(self):
        """
        Compute betweenness centrality for each node using a modified Brandes' algorithm.

        Args: None

        Returns:
               betweenness: dictionary with betweenness centrality scores for each node.
        """
        # Initialize the centrality dictionary with zero for each node
        betweenness = {node: 0 for node in self.nodes}

        # Loop through each node in the graph as the source node 's'
        for source in self.nodes:
            # Use shortest_paths to calculate distances and predecessors
            distances, predecessors = self.shortest_paths(source)

            # Stack to store nodes in reverse order for processing
            stack = []
            for node in sorted(self.nodes, key=lambda x: distances[x], reverse=True):
                stack.append(node)

            # Initialize a dictionary for dependency propagation
            dependency = {node: 0 for node in self.nodes}

            # Process the nodes in reverse order (post-order)
            while stack:
                node = stack.pop()
                for pred in predecessors[node]:
                    # Update the dependency of the predecessor based on the current node
                    # This is a modification of the classic formula to account for both the distance of the current node 
                    # and the number of predecessors. The formula adjusts the dependency by incorporating the distance of the node 
                    # and the number of predecessors, scaling it with a factor of 0.07. If the current node distance is zero, no update occurs.
                    dependency[pred] += (1+dependency[node]) / (len(predecessors[node]) * (0.07*distances[node])) if distances[node]!=0 else 0
                    if pred != source:
                        betweenness[pred] += dependency[pred]

        # Calculate the number of possible combinations of 2 nodes from a set of len(self.nodes) nodes
        # The correct formula for C(n, 2) is (n * (n - 1)) / 2
        normalization = (len(self.nodes) - 1) * (len(self.nodes) - 2) / 2
        for node in betweenness:
            betweenness[node] /= normalization # Normalize the betweenness centrality 

        return betweenness
    
    def closeness_centrality(self, node:str):
        """
        Compute closeness centrality for a specific node.
        
        Args: 
            node: Node to compute centrality 

        Returns: 
               Closeness centrality score
        """
        # Compute distances from node to all other nodes
        distances, _ = self.shortest_paths(node)
        
        # Remove infinite distances and compute Closeness
        reachable_distances = [d for d in distances.values() if d != float('inf') and d > 0]
        
        if not reachable_distances:
            return 0
        
        # Closeness is inverse of average distance
        return (len(reachable_distances)) / sum(reachable_distances)
    
    def degree_centrality(self, node:str):
        """
        Compute degree centrality (number of direct connections).
        
        Args:
            node: Node to compute centrality 

        Returns: 
               Degree centrality score
        """
        # Maximum number of connections possible in an undirected graph
        n = len(self.graph) - 1 # Excludes the node itself
        
        return len(self.graph[node])/n
    
    def pagerank(self, p=0.85, epsilon= 0.001, max_iterations=200):
        """
        Compute PageRank for all nodes in the network.
        
        Args:
            p: Probability of following a link
            epsilon: Convergence threshold
            max_iterations: Maximum number of iterations

        Returns: 
               pagerank_scores: dictionary of PageRank scores
        """
        # Initialize PageRank
        num_nodes = len(self.nodes)
        pagerank_scores = {node: 1/num_nodes for node in self.nodes}
        
        for _ in range(max_iterations):
            prev_pagerank = pagerank_scores.copy()
            
            # Compute new PageRank for each node
            for node in self.nodes:
                link_contribution  = 0
                for other_node in self.nodes:
                    if node in self.graph[other_node]:
                        # Contribution proportional to outgoing links
                        out_degree = len(self.graph[other_node])
                        link_contribution  += prev_pagerank[other_node] / out_degree
                
                # Apply damping factor
                # (1 - p) / num_nodes represents the probability of "random jump" or teleportation. 
                # It indicates the probability that a user will decide to go to another page at random.
                pagerank_scores[node] = (1 - p) / num_nodes + p * link_contribution 
            
            # Check convergence
            if all(abs(pagerank_scores[node] - prev_pagerank[node]) < epsilon for node in self.nodes):
                break
        
        return pagerank_scores

    def katz_centrality(self, alpha=0.005, threshold=0.001, max_iterations=200):
        """
        Calculate Katz centrality for nodes in a graph with convergence check.
        Args:
            alpha: The decay parameter
            threshold: Convergence threshold
            max_iterations: Maximum number of iterations
        Returns:
               Dictionary of katz centrality scores
        """
        num_nodes = len(self.nodes)
        centrality = {node: 1 / num_nodes for node in self.nodes}

        for _ in range(max_iterations):
            prev_centrality = centrality.copy()
            for node in self.nodes:
                centrality[node] = alpha * sum(prev_centrality[neighbor] for neighbor in self.graph[node]) + (1 - alpha) / num_nodes
            
            # Normalize (using sum of squares for better scaling)
            norm = sum(c**2 for c in centrality.values())**0.5
            centrality = {node: c / norm for node, c in centrality.items()}
            
            # Convergence check
            if max(abs(centrality[node] - prev_centrality[node]) for node in self.nodes) < threshold:
                break

        return centrality

#------------------------------------------------------------------------------------#  

    def analyze_centrality(self, airport:str):
        """
        Comprehensive centrality analysis for a specific node.
        This function compute the following centrality measures for a given airport:
        - Betweenness Centrality
        - Closeness Centrality
        - Degree Centrality
        - Pagerank Score

        Args:
            airport: Node to analyze

        Returns:
               DataFrame of centrality measures
        """
        return pd.DataFrame({
            "Airport": [airport],
            'Betweenness Centrality': [self.betweenness_centrality()[airport]],
            'Closeness Centrality': [self.closeness_centrality(airport)],
            'Degree Centrality': [self.degree_centrality(airport)],
            'PageRank Score': [self.pagerank()[airport]]})
