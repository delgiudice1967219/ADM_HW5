import pandas as pd
import networkx as nx
import heapq

class FlightNetwork:
    """
A class to represent and analyze a flight network, where airports are nodes and flights are directed edges 
with weights corresponding to the distance between airports. The class allows adding flights, computing 
shortest paths using Dijkstra's algorithm, and finding the best routes between airports for a given date.

Methods:

add_flight(origin, destination, distance): Adds a flight between two airports to the network.

shortest_paths(start): Computes the shortest paths from a starting airport to all other airports using Dijkstra's algorithm.

find_best_routes(origin_city, destination_city, date, flights): Finds the best routes between origin and 
destination cities for the given date based on available flights.

Functions: 

information_collection(flights):

Allows the user to select a departure city, destination city and flight date from a DataFrame of flights. 
Filters the data based on the user's choices and returns the selected information (Origin city name, Destination city name and considered date).
    """

    def __init__(self):
        """
        Initialize the network analyzer with an empty directed graph and an empty set of nodes.

        This constructor initializes:
            A directed graph (DiGraph) to represent the network of airports and flights.
            A set to store all unique nodes (airports) in the network.
        """
        # Initialize an empty directed graph to store flights between airports
        self.graph = nx.DiGraph()

        # Set to store all nodes (airports) in the graph for easy access
        self.nodes = set()

#------------------------------------------------------------------------------------#  

    def add_flight(self, origin, destination, distance):
        """
        Add a flight to the network graph.

        Args:
            origin (str): Origin airport.
            destination (str): Destination airport.
            distance (float): Distance of the flight.
        """
        # Ensure the origin and destination airports are added as nodes
        self.nodes.add(origin)
        self.nodes.add(destination)
        
        # Add the flight to the graph
        self.graph.add_edge(origin, destination, weight=distance)

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
        previous_nodes = {node: None for node in self.nodes}
        unvisited = [(0, start)]   # List of tuples (distance, node)

        while unvisited:
            current_distance, current_node = heapq.heappop(unvisited) 
            # If a node has already been processed, skip it
            if current_distance > distances[current_node]:
                continue

             # Check all neighbors
            for neighbor, edge_attributes  in self.graph[current_node].items():
                weight = edge_attributes ['weight']
                distance = current_distance + weight

                # If a shorter path is found
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(unvisited, (distance, neighbor)) 

        return distances, previous_nodes
    
#------------------------------------------------------------------------------------#  

    def find_best_routes(self, origin_city: str, destination_city: str, date, flights: pd.DataFrame):
        """
        Find the best routes between cities for the given date.

        Args:
            origin_city (str): Origin city.
            destination_city (str): Destination city.
            date (datetime): Flight date in yyyy-mm-dd format.
            flights (DataFrame): DataFrame of flight dictionaries.

        Returns:
               DataFrame: A DataFrame containing the best routes between airports.
        """
        # Filter flights based on the given date
        date_flights = flights[flights['Fly_date'] == date]
        
        # Add all flights to the network
        for _, flight in date_flights.iterrows():
            self.add_flight(flight['Origin_airport'],flight['Destination_airport'],flight['Distance'])
        
        # Get unique origin and destination airports
        origin_airports = date_flights[date_flights['Origin_city'] == origin_city]['Origin_airport'].unique()
        
        destination_airports = date_flights[date_flights['Destination_city'] == destination_city]['Destination_airport'].unique()

        best_path = []
        
        # Find best routes for all airport pairs
        for origin_airport in origin_airports:
            distances, previous_nodes = self.shortest_paths(origin_airport)
            
            for destination in destination_airports:
                if destination not in distances or distances[destination] == float('inf'):
                    best_path.append({'Origin_city_airport': origin_airport,
                        'Destination_city_airport': destination,
                        'Best_route': 'No route found'})
                else:
                    # Reconstruct the path
                    path = []
                    current = destination
                    while current is not None:
                        path.append(current)
                        current = previous_nodes[current]
                    path.reverse()
                    
                    best_path.append({'Origin_city_airport': origin_airport,
                        'Destination_city_airport': destination,
                        'Best_route': ' → '.join(path)})

        return pd.DataFrame(best_path)
    
#------------------------------------------------------------------------------------#  

def information_collection(flights: pd.DataFrame):
    """
    Filters flight information based on user input for departure city, destination city, and flight date.

    Args:
        flights (pd.DataFrame): A DataFrame containing flight information with columns such as 
                                'Origin_city', 'Destination_city', and 'Fly_date'.

    Returns:
           tuple: A tuple containing the selected origin city, destination city, and flight date (str).
    
    Raises:
        TypeError: If the departure city or destination city is not found.
        ValueError: If no flights are found for the given date.
    """
    # Prompt the user to select the departure city
    origin_city = input("Select the departure city (in English): ").title()
    if origin_city in flights["Origin_city"].values:
        # Filter flights based on the selected departure city
        filter_flights = flights.loc[flights["Origin_city"] == origin_city]

        print("origin_city: ", origin_city)
        print("List of destination city availables:", filter_flights["Destination_city"].unique())

        # Prompt the user to select the destination city
        destination_city = input("Select the destination city (in English): ").title()
        filter_flights = filter_flights.loc[filter_flights["Destination_city"] == destination_city]
        print("Available dates: ", *filter_flights["Fly_date"].unique())
        
        # If there is only one available date, return it directly
        if len(filter_flights["Fly_date"]) == 1:
            date = filter_flights["Fly_date"].iloc[0]  # Safely access the single date
            print(f"You selected the following information:\n"
                f"Departure City: {origin_city}\n"
                f"Arrival City: {destination_city}\n"
                f"Selected Date: {date}")
            return origin_city, destination_city, date
        
        # Prompt the user to enter a specific flight date
        date = input("Enter the flight date (in format YYYY-MM-DD): ")
        try:
            # Validate and filter by the selected date
            filter_flights = filter_flights.loc[filter_flights["Fly_date"] == date]
            if filter_flights.empty:
                raise ValueError("No flights found for the given date.")
            else:
                print(f"You selected the following information:\n"
                f"Departure City: {origin_city}\n"
                f"Arrival City: {destination_city}\n"
                f"Selected Date: {date}")
                return origin_city, destination_city, date
        except ValueError as e:
            # Handle invalid date or no flights found
            print(f"Error: {e}")
    else:
        # Handle invalid departure city input
        raise TypeError("Sorry, departure city not found. Please retry with a different city.")
