import folium
import pickle
import os


def plot_communities(partition, cities_to_mark=None):
    """
    Plot cities on a Folium map with different colors based on community assignments.
    Each city is plotted once, with its community-based marker, and optionally,
    additional markers can be added for specified cities.
    
    Parameters:
    - partition: dict
        The community partition of cities where keys are city names and values are community ids.
    - cities_to_mark: list, optional
        A list of city names to mark explicitly on the map with a special marker. 
        If not provided, no additional markers will be added for cities.
    
    Returns:
    - folium.Map object
        The Folium map with cities and community-based markers, plus additional markers for the specified cities.
    """

    # Default center of the map (centered on the U.S.) and zoom level
    map_center = (39.8283, -98.5795)
    zoom_start = 4

    # Create a folium map centered at the given location
    m = folium.Map(location=map_center, zoom_start=zoom_start)

    # Load dictionary with city coordinates
    file_path = os.path.join(os.path.dirname(__file__), 'city_coordinates.pkl')
    with open(file_path, 'rb') as f:
        city_coordinates = pickle.load(f)

    # Define a color palette for communities
    community_colors = {
        community: f'#{(community * 33) % 256:02x}{(community * 66) % 256:02x}{(community * 99) % 256:02x}'
        for community in set(partition.values())
    }

    # Iterate over each city in the coordinates dictionary and add a marker
    for city, (lat, lon) in city_coordinates.items():
        # Get the community for the city
        community = partition.get(city)

        # Choose a color based on the community assignment
        color = community_colors.get(community, '#000000')  # Default to black if no color is found

        # Add a marker to the map for the city
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=f'{city} - Community {community}'
        ).add_to(m)

    # If a list of cities to mark is provided, add additional markers for those cities
    if cities_to_mark:
        for city in cities_to_mark:
            # Check if the city exists in the coordinates dictionary
            if city in city_coordinates:
                lat, lon = city_coordinates[city]
                # Add a special marker for the city
                folium.Marker(
                    location=[lat, lon],
                    popup=f'{city}',
                    icon=folium.Icon(color='red', icon='star')
                ).add_to(m)
            else:
                print(f"City {city} not found in the coordinates data.")

    return m
