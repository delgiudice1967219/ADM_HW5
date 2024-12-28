from pyspark.sql import SparkSession
from operator import add
import time
from collections import defaultdict


def connected_components(flight_network, start_date, end_date, show=False):
    """
    Analyzes a flight network to find connected components in a given date range.
    
    Args:
        flight_network (pd.DataFrame): A DataFrame containing flight data with columns:
            - "Fly_date" (str): The flight date.
            - "Origin_airport" (str): The origin airport.
            - "Destination_airport" (str): The destination airport.
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.
        show (bool): If True, prints the execution time.
    
    Returns:
        tuple:
            - Number of connected components.
            - A dictionary with component sizes (key: component_id, value: size).
            - A list of airports in the largest connected component.
    """
    # Start timer
    start_time = time.time()
    
    # Create SparkSession
    spark = SparkSession.builder.appName("FlightNetworkAnalysis").getOrCreate()
    
    # Filter flight data within the date range using Pandas
    filtered_data = flight_network[
        (flight_network["Fly_date"] >= start_date) & (flight_network["Fly_date"] <= end_date)
    ]
    
    # Extract edges
    edges = filtered_data[["Origin_airport", "Destination_airport"]].drop_duplicates()
    edges_list = list(edges.itertuples(index=False, name=None))
    # Make sure the graph is undirected
    edges_list += [(dst, src) for src, dst in edges_list]  # Add reversed edges
    edges_list = list(set(edges_list))  # Remove duplicates
    
    if not edges_list:
        print("No flight found within the specified date range.")
        return 0, {}, []

    # Convert edge list to Spark RDD
    edges_rdd = spark.sparkContext.parallelize(edges_list)
    
    # Compute connected components
    result = find_connected_components(edges_rdd, show=show)
    final_components = result.collect()
    
    # Organize vertices by component
    component_to_vertices = defaultdict(list)
    for vertex, component in final_components:
        component_to_vertices[component].append(vertex)
    
    # Sort components by size (largest to smallest) and reindex
    sorted_components = sorted(component_to_vertices.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Reindex components
    indexed_components = {i: vertices for i, (component, vertices) in enumerate(sorted_components)}
    
    # Analyze components
    component_sizes = {i: len(vertices) for i, vertices in indexed_components.items()}
    largest_component_airports = sorted(indexed_components[0])  # Airports in the largest component
    
    # End timer
    end_time = time.time()
    
    # Optionally show execution time
    if show:
        print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    # Return results
    return len(indexed_components), component_sizes, largest_component_airports



def initialize_vertices(edges, spark):
    """
    Initialize each vertex with its own ID as its component ID
    Input format: RDD of (src, dst) pairs
    """
    # Get all unique vertices
    vertices = edges.flatMap(lambda x: [x[0], x[1]]).distinct()
    # Initialize each vertex with itself as component ID
    return vertices.map(lambda x: (x, x))

def generate_messages(vertex):
    """
    For each vertex, generate messages to be sent to neighbors
    Input: (vertex_id, component_id)
    Output: [(neighbor_id, component_id)]
    """
    vertex_id, component_id = vertex
    return [(vertex_id, component_id)]

def propagate_min(edges, vertices):
    """
    Propagate minimum component ID to neighbors
    """
    # Join edges with current component IDs
    edge_with_components = edges.join(vertices)
    
    # Generate messages for each edge
    messages = edge_with_components.flatMap(
        lambda x: [(x[1][0], x[1][1]), (x[0], x[1][1])]
    )
    
    # Combine messages by taking minimum component ID
    new_components = messages.reduceByKey(min)
    
    return new_components

def find_connected_components(edges, max_iterations=50, show=True):
    """
    Find connected components using iterative MapReduce
    
    Args:
        edges: RDD of (src, dst) pairs representing graph edges
        max_iterations: Maximum number of iterations to run
    
    Returns:
        RDD of (vertex_id, component_id) pairs
    """
    # Create SparkSession
    spark = SparkSession.builder.appName("ConnectedComponents").getOrCreate()
    
    # Initialize vertices with their own IDs as component IDs
    current_components = initialize_vertices(edges, spark)
    
    # Iterate until convergence or max iterations
    for i in range(max_iterations):
        if show:
            print(f'Iteration: {i}')
        # Save old components for convergence check
        old_components = current_components
        
        # Propagate minimum component IDs
        current_components = propagate_min(edges, current_components)
        
        # Check for convergence (no changes in component IDs)
        hash_current = current_components.map(lambda x: hash(x)).reduce(add)
        hash_old = old_components.map(lambda x: hash(x)).reduce(add)

        if hash_current == hash_old:
            break
    
    return current_components


