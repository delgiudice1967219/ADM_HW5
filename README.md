# ADM - Homework 5: USA Airport Flight Analysis, Group #10

This GitHub repository contains the implementation of the fifth homework assignment for the **Algorithmic Methods of Data Mining** course (2024-2025) for the master's degree in Data Science at Sapienza University. This project focuses on analyzing the USA Airport Dataset and implementing a series of algorithms to explore flight networks, assess centrality measures, find optimal routes, and detect communities. The details of the assignement are specified [here](https://github.com/Sapienza-University-Rome/ADM/tree/master/2024/Homework_5).

**Team Members:**
- Xavier Del Giudice, 1967219, delgiudice.1967219@studenti.uniroma1.it
- Flavio Mangione, 2201201, flaviomangio@gmail.com
- Leonardo Rocci, 1922496, rocci.1922496@studenti.uniroma1.it
- Andrea Di Vincenzo, 1887012, divincenzo.1887012@studenti.uniroma1.it

**Main Notebook:**  
The main script can be visualized using [nbviewer]().

---

## Repository Structure

```plaintext
├── us-states.json               # GeoJSON file with USA map data for visualization
├── functions/                   # Directory containing core project modules
│   ├── flight_network.py        # Module for graph-based analysis
│   ├── map_visualizer.py        # Module for create an interactive map visualization
│   ├── partitioner.py           # Module for create partition of the graph and visualize them
│   ├── AirlineRouteNetwork.py   # Module for managing and analyzing flight networks
│   ├── Centrality_Graph.py      # Module for compute and compare centrality measures for all nodes in the graph
│   ├── connected_components.py  # Module to compute connected components within a graph
│   └── Network_Metrics.py       # Module for computing and analyzing various graph centrality metrics 
├── Communities/                 # Package with modules related to Community Detection
│   ├── __init__.py              # Initializes the package
│   ├── community.py             # Contains a class used within the Louvain algorithm for community detection
│   ├── louvain.py               # Implements the Louvain algorithm as a class
│   ├── optimized_louvain.py     # Provides a vectorized version of the Louvain algorithm implemented as a function
│   ├── plot.py                  # Contains a function to plot detected communities in a map
│   ├── city_graph.pkl           # A graph of cities extracted from a dataset, with edge weights representing flights between cities
│   └── city_coordinates.pkl     # A dictionary mapping each city to its geographical coordinates
├── main.ipynb                  # Main notebook with the implementation and results
├── .gitignore                  # Specifies files and directories ignored by Git
├── README.md                   # Project documentation
└── LICENSE                     # License file for the project
```

Here are links to all the files:

* [us-states.json](us-states.json): GeoJSON file containing the USA map.
* [functions](functions/): Contains Python modules with reusable functions for each analysis task.
  * [flight_network.py](functions/flight_network.py) Module for graph-based analysis.
  * [map_visualizer.py](functions/map_visualizer.py): Module for create an interactive map visualization using dash.
  * [partitioner.py](functions/partitioner.py): Module for create two partition of the original graph and visualize them on the USA map. 
  * [connected_components.py](functions/connected_components.py): Module to compute connected components within a graph.
  * [AirlineRouteNetwork.py](functions/AirlineRouteNetwork.py): Module for analyzing flight networks with route planning and shortest path computations between airports.
  * [Centrality_Graph.py](functions/Centrality_Graph.py): Module for compute and compare centrality measures for all nodes in the graph.
  * [Network_Metrics.py](functions/Network_Metrics.py): Module for computing and analyzing various graph centrality metrics. 
* [Communities](Communities/): Package with modules related to Community Detection.
  * [__init__.py](Communities/__init__.py) Module that initializes the package.
  * [community.py](Communities/community.py): Module that contains the implementation within the Louvain algorithm for community detection.
  * [louvain.py](Communities/louvain.py): Module that implements the Louvain algorithm. 
  * [optimized_louvain.py](Communities/optimized_louvain.py): Module that provides a vectorized implementation of the Louvain algorithm.
  * [plot.py](Communities/plot.py): Module that contains functions to plot detected communities in the USA map.
  * [city_graph.pkl](Communities/city_graph.pkl): Pickle file that store a graph of cities extracted from the dataset.
  * [city_coordinates.pkl](Communities/city_coordinates.pkl): Pickle file storing a dictionary that maps each city to its geographical coordinates.
* [main.ipynb](main.ipynb): The main notebook presenting the tasks, solutions, and results.  
* [README.md](README.md): Project documentation.  
* LICENSE: License file for the project.

---

## Project Overview

This project leverages network analysis, optimization techniques, and visualization tools to explore the USA flight network, extract meaningful insights, and propose solutions to real-world problems.

### 1. Flight Network Analysis (Q1)
We analyze the basic features of the USA flight network graph, including size, density, and degree distribution. Key outputs include:  
- Graph metrics (nodes, edges, density)  
- Identification of hub airports  
- Busiest routes by passenger flow and efficiency
- Under and Over-Utilized routes
- Interactive flight network map: the map is developed in a Dash app, so it is not visible in the notebook. To use it, you need to run the respective cell and click on the generated link. Below is an example of how it works.

#### Usage of the interactive map
https://github.com/user-attachments/assets/173a720a-48e3-4a25-b638-9f7945798eb9

**Key Technologies Used:**  
- Python: Pandas, Matplotlib, Networkx, Dash, Folium  

### 2. Nodes' Contribution (Q2)
Using centrality measures, we identify critical airports in the network that play a pivotal role in connectivity. Outputs include:  
- Centrality measures: Betweenness, closeness, degree, PageRank  
- Histograms for centrality distributions  
- Top airports for each centrality measure

**Key Technologies Used:**  
- Python: networkx, pandas, matplotlib, seaborn 

### 3. Finding Best Routes (Q3)
We implement a shortest-path algorithm to find the optimal route between two cities based on flight distances. The function:  
- Accounts for multiple airports per city  
- Outputs the best route for each airport pair  
- Handles cases where no route exists

**Key Technologies Used:**  
- Python: networkx, custom algorithms  

### 4. Airline Network Partitioning (Q4)
We address the graph disconnection problem by removing the minimum number of flights to separate the network into two disconnected subgraphs. Outputs include:  
- Flights removed to achieve separation  
- Visualizations of the original and partitioned networks on the USA map

**Key Technologies Used:**  
- Python: Networkx, Matplotlib, GeoPandas   

### 5. Finding and Extracting Communities (Q5)
Using community detection algorithms, we analyze city-level flight connectivity. Key deliverables include:  
- Total communities and their composition  
- Visualizations highlighting detected communities  
- Identification of cities belonging to the same community  

**Key Technologies Used:**  
- Python: networkx, matplotlib, community (Louvain method)  

### Bonus: Connected Components on MapReduce
We implement a PySpark-based MapReduce algorithm to identify connected components in the flight network over a specific period. Key outputs include:  
- Number and size of connected components  
- Airports in the largest connected component  
- Comparison with GraphFrames package

**Key Technologies Used:**  
- PySpark  

### Algorithmic Question (AQ)
We solve an optimization problem to find the cheapest flight route under specific constraints (maximum stops). Key outputs include:  
- Cost for the optimal path  
- Efficiency analysis and scalability improvements  

**Key Technologies Used:**  
- Python: priority queue, custom optimization algorithms

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
