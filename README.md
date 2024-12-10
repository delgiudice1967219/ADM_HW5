# ADM - Homework 5: USA Airport Flight Analysis, Group #10

This GitHub repository contains the implementation of the fifth homework assignment for the **Algorithmic Methods of Data Mining** course (2024-2025) for the master's degree in Data Science at Sapienza University. This project focuses on analyzing the USA Airport Dataset and implementing a series of algorithms to explore flight networks, assess centrality measures, find optimal routes, and detect communities.

**Team Members:**
- Xavier Del Giudice, 1967219, delgiudice.1967219@studenti.uniroma1.it
- [Your Name], [Your Matricola], [Your Email]
- [Your Name], [Your Matricola], [Your Email]

**Main Notebook:**  
The main script can be visualized using [nbviewer]().

---

## Repository Structure

```plaintext
├── datasets/                   # Directory containing the dataset files
│   ├── USA_airports.csv         # Airport data
│   ├── flight_routes.csv        # Flight routes data
│   └── passengers.csv           # Passenger data
├── functions/                  # Directory containing core project modules
│   ├── graph_analysis.py        # Module for graph-based analyses (e.g., degree, density)
│   ├── centrality_measures.py   # Module for calculating centrality measures
│   ├── route_finder.py          # Module for finding optimal routes
│   ├── partitioning.py          # Module for graph partitioning
│   ├── community_detection.py   # Module for community detection
│   └── utils.py                 # Helper functions for data processing and visualization
├── main.ipynb                  # Main notebook with the implementation and results
├── .gitignore                  # Specifies files and directories ignored by Git
├── README.md                   # Project documentation
└── LICENSE                     # License file for the project
```

### File Descriptions

- **[datasets](datasets/):** Contains the dataset files used in the analysis.  
- **[functions](functions/):** Contains Python modules with reusable functions for each analysis task.  
- **[main.ipynb](main.ipynb):** The main notebook presenting the tasks, solutions, and results.  
- **[README.md](README.md):** Project documentation (this file).  
- **LICENSE:** Specifies the licensing details for this project.

---

## Project Overview

This project leverages network analysis, optimization techniques, and visualization tools to explore the USA flight network, extract meaningful insights, and propose solutions to real-world problems.

### 1. Flight Network Analysis (Q1)
We analyze the basic features of the USA flight network graph, including size, density, and degree distribution. Key outputs include:  
- Graph metrics (nodes, edges, density)  
- Identification of hub airports  
- Busiest routes by passenger flow and efficiency  
- Interactive flight network map

**Key Technologies Used:**  
- Python: pandas, matplotlib, seaborn, networkx, plotly  

### 2. Nodes' Contribution (Q2)
Using centrality measures, we identify critical airports in the network that play a pivotal role in connectivity. Outputs include:  
- Centrality measures: Betweenness, closeness, degree, PageRank  
- Histograms for centrality distributions  
- Top airports for each centrality measure

**Key Technologies Used:**  
- Python: networkx, matplotlib  

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
- Visualizations of the original and partitioned networks  

**Key Technologies Used:**  
- Python: networkx, matplotlib  

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
- Cost and route for the optimal path  
- Efficiency analysis and scalability improvements  

**Key Technologies Used:**  
- Python: priority queue, custom optimization algorithms  

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Contact

For any questions or collaborations, please reach out to the team members via the emails provided above.  
