import matplotlib.pyplot as plt
from Network_Metrics import GraphMetrics
import pandas as pd
import seaborn as sns

#--------------------------------PLOTS V1--------------------------------#  

def compare_centralities(flight_network):
    """
    Compute and compare centrality measures for all nodes in the graph. 
    For the first version of this functions, there are four Centrality Measures:
    - Betweenness Centrality
    - Closeness Centrality
    - Degree Centrality
    - Pagerank 

    Args:
        flight_network (nx.DiGraph): A directed networkx graph representing the flight network.

    Returns:
           Table: Top 5 airports for each centrality measure.
    """

    # Initialize GraphMetrics instance
    G = GraphMetrics(flight_network)

    # Create dictionaries to store centrality scores calculated using different measures
    betweenness_scores = G.betweenness_centrality()
    degree_scores = {}
    closeness_scores = {}
    pagerank_scores = G.pagerank()

    # Calculate centrality measures for each node (airport)
    for node in G.nodes:
        degree_scores[node] = G.degree_centrality(node)
        closeness_scores[node] = G.closeness_centrality(node)

    # Sorting and selecting top 5 airports for each centrality measure
    top_betweenness = sorted(betweenness_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    top_degree = sorted(degree_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    top_closeness = sorted(closeness_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    top_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Creating a DataFrame to display the results in tabular format
    table = pd.DataFrame({
        'Centrality Measure': ['Betweenness'] * 5 + ['Degree'] * 5 + ['Closeness'] * 5 + ['PageRank'] * 5 ,
        'Airport': [x[0] for x in top_betweenness] + [x[0] for x in top_degree] +
                   [x[0] for x in top_closeness] + [x[0] for x in top_pagerank],
        'Score': [x[1] for x in top_betweenness] + [x[1] for x in top_degree] +
                 [x[1] for x in top_closeness] + [x[1] for x in top_pagerank]})

    # Plotting the distributions of centrality measures
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 2, 1)
    sns.histplot(list(betweenness_scores.values()), bins=50, color="#F72585", edgecolor="black")
    plt.title('Betweenness Centrality Distribution', weight="bold")
    plt.xlabel('Betweenness Centrality')
    plt.ylabel('Frequency')
    plt.grid(linestyle='--', linewidth=0.5, color="black", axis='y')
    
    plt.subplot(2, 2, 2)
    sns.histplot(list(degree_scores.values()), bins=50, color="#7209B7", edgecolor="black")
    plt.title('Degree Centrality Distribution', weight="bold")
    plt.xlabel('Degree Centrality')
    plt.ylabel('Frequency')
    plt.grid(linestyle='--', linewidth=0.5, color="black", axis='y')

    plt.subplot(2, 2, 3)
    sns.histplot(list(closeness_scores.values()), bins=50, color="#3A0CA3", edgecolor="black")
    plt.title('Closeness Centrality Distribution', weight="bold")
    plt.xlabel('Closeness Centrality')
    plt.ylabel('Frequency')
    plt.grid(linestyle='--', linewidth=0.5, color="black", axis='y')

    plt.subplot(2, 2, 4)
    sns.histplot(list(pagerank_scores.values()), bins=50, color="#4361EE", edgecolor="black")
    plt.title('PageRank Score Distribution', weight="bold")
    plt.xlabel('PageRank Score')
    plt.ylabel('Frequency')
    plt.grid(linestyle='--', linewidth=0.5, color="black", axis='y')

    plt.tight_layout(pad=2.0)
    plt.show()

    return table

#--------------------------------PLOTS V2--------------------------------#  

def compare_centralities_v2(flight_network):
    """
    Compute and compare centrality measures for all nodes in the graph.
    For the second version of this functions, there are five Centrality Measures:
    - Betweenness Centrality
    - Closeness Centrality
    - Degree Centrality
    - Pagerank 
    - Katz Centrality 

    Args:
        flight_network (nx.DiGraph): A directed networkx graph representing the flight network.
        
    Returns:
           Table: Top 5 airports for each centrality measure
    """

    # Initialize GraphMetrics instance
    G = GraphMetrics(flight_network)
    
    # Create dictionaries to store centrality scores calculated using different measures
    betweenness_scores = G.betweenness_centrality()
    degree_scores = {}
    closeness_scores = {}
    pagerank_scores = G.pagerank()
    katz_scores = G.katz_centrality()

    # Calculate centrality measures for each node (airport)
    for node in G.nodes:
        degree_scores[node] = G.degree_centrality(node)
        closeness_scores[node] = G.closeness_centrality(node)

    # Sorting and selecting top 5 airports for each centrality measure
    top_betweenness = sorted(betweenness_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    top_degree = sorted(degree_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    top_closeness = sorted(closeness_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    top_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    top_katz= sorted(katz_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    # Creating a DataFrame to display the results in tabular format
    table = pd.DataFrame({
        'Centrality Measure': ['Betweenness'] * 5 + ['Degree'] * 5 + ['Closeness'] * 5 + ['PageRank'] * 5 + ['Katz'] * 5,
        'Airport': [x[0] for x in top_betweenness] + [x[0] for x in top_degree] +
                   [x[0] for x in top_closeness] + [x[0] for x in top_pagerank] + [x[0] for x in top_katz],
        'Score': [x[1] for x in top_betweenness] + [x[1] for x in top_degree] +
                 [x[1] for x in top_closeness] + [x[1] for x in top_pagerank] + [x[1] for x in top_katz]})

    # Plotting the distributions of centrality measures
    plt.figure(figsize=(15, 8))

    plt.subplot(3, 2, 1)
    sns.histplot(list(betweenness_scores.values()), bins=50, color="#F72585", edgecolor="black")
    plt.title('Betweenness Centrality Distribution', weight="bold")
    plt.xlabel('Betweenness Centrality')
    plt.ylabel('Frequency')
    plt.grid(linestyle='--', linewidth=0.5, color="black", axis='y')

    plt.subplot(3, 2, 2)
    sns.histplot(list(degree_scores.values()), bins=50, color="#7209B7", edgecolor="black")
    plt.title('Degree Centrality Distribution', weight="bold")
    plt.xlabel('Degree Centrality')
    plt.ylabel('Frequency')
    plt.grid(linestyle='--', linewidth=0.5, color="black", axis='y')

    plt.subplot(3, 2, 3)
    sns.histplot(list(closeness_scores.values()), bins=50, color="#3A0CA3", edgecolor="black")
    plt.title('Closeness Centrality Distribution', weight="bold")
    plt.xlabel('Closeness Centrality')
    plt.ylabel('Frequency')
    plt.grid(linestyle='--', linewidth=0.5, color="black", axis='y')

    plt.subplot(3, 2, 4)
    sns.histplot(list(pagerank_scores.values()), bins=50, color="#4361EE", edgecolor="black")
    plt.title('PageRank Score Distribution', weight="bold")
    plt.xlabel('PageRank Score')
    plt.ylabel('Frequency')
    plt.grid(linestyle='--', linewidth=0.5, color="black", axis='y')

    plt.subplot(3, 2, 5)
    sns.histplot(list(katz_scores.values()), bins=50, color="#4CF0B9", edgecolor="black")
    plt.title('Katz Centrality Distribution', weight="bold")
    plt.xlabel('Katz Centrality')
    plt.ylabel('Frequency')
    plt.grid(linestyle='--', linewidth=0.5, color="black", axis='y')

    plt.tight_layout()
    plt.show()

    return table
