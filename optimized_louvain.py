import numpy as np
import networkx as nx


def louvain_vectorized(graph, max_iter=1000):
    """
    Perform community detection using a vectorized Louvain-like method.

    Parameters:
    - graph: networkx.Graph
        The graph object (directed or undirected).
    - max_iter: int
        Maximum number of iterations for convergence.

    Returns:
    - dict
        A dictionary where keys are city names (from the graph) and values are their assigned communities.
    """

    # Convert directed graph to undirected if necessary
    if graph.is_directed():
        graph = graph.to_undirected()

    # Create the adjacency matrix
    adj_matrix = nx.to_numpy_array(graph, nodelist=list(graph.nodes()), dtype=float, weight='weight')

    # Initialize variables
    rows = adj_matrix.shape[0]
    communities = np.arange(rows)  # Each node starts in its own community
    k = adj_matrix.sum(axis=1)  # Degree of each node
    total_weight = k.sum()
    inv_m = 1.0 / (2.0 * total_weight)  # 1 / (2 * total weight)

    vertices_changed = rows

    for _ in range(max_iter):
        if vertices_changed == 0:
            break

        vertices_changed = 0

        for j in range(rows):
            # Extract neighbors and their community assignments
            neighbors = np.nonzero(adj_matrix[j])[0]
            weights = adj_matrix[j, neighbors]

            # Calculate modularity gain for all possible moves
            current_comm = communities[j]
            neighbor_comms = communities[neighbors]
            modularity_gain = weights - k[j] * k[neighbors] * inv_m

            # Aggregate modularity gain per community
            unique_comms = np.unique(neighbor_comms)
            community_modularity = np.zeros(len(unique_comms))
            for idx, comm in enumerate(unique_comms):
                community_modularity[idx] = modularity_gain[neighbor_comms == comm].sum()

            # Check if there are communities to move to
            if len(unique_comms) > 0:  # Ensure there are communities to select from
                # Select the best community (maximize modularity gain)
                best_comm_idx = np.argmax(community_modularity)
                best_comm = unique_comms[best_comm_idx]

                if best_comm != current_comm:
                    communities[j] = best_comm
                    vertices_changed += 1

    # Normalize community ids to be continuous from 0 to n-1
    unique_communities = np.unique(communities)
    community_mapping = {comm: idx for idx, comm in enumerate(unique_communities)}
    normalized_communities = np.array([community_mapping[comm] for comm in communities])

    # Get the city names from the graph
    city_names = list(graph.nodes())

    # Return the mapping from city name to community
    city_community_mapping = {city_names[idx]: normalized_communities[idx] for idx in range(rows)}

    return city_community_mapping
