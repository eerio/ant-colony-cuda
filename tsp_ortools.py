import sys
import math
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

def parse_tsp_file(filename):
    """Parse TSPLIB format file and return coordinates and edge weight type."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    coords = []
    coord_section = False
    edge_weight_type = None

    for line in lines:
        line = line.strip()

        if line.startswith('EDGE_WEIGHT_TYPE'):
            edge_weight_type = line.split(':')[1].strip().upper()
            if edge_weight_type not in ('EUC_2D', 'CEIL_2D', 'GEO'):
                raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}")

        if line.startswith('NODE_COORD_SECTION'):
            coord_section = True
            continue

        if line.startswith('EOF'):
            break

        if coord_section:
            parts = line.split()
            if len(parts) >= 3:
                _, x, y = parts
                coords.append((float(x), float(y)))

    if edge_weight_type is None:
        raise ValueError("EDGE_WEIGHT_TYPE not found in the .tsp file!")

    return edge_weight_type, coords

def geo_to_degrees(coord):
    """Convert GEO TSPLIB format to decimal degrees."""
    deg = int(coord)
    min_ = coord - deg
    return deg + (min_ * 100) / 60

def preprocess_coords(edge_weight_type, coords):
    """Preprocess coordinates based on edge weight type."""
    if edge_weight_type == 'GEO':
        # Convert each (longitude, latitude) properly
        return [(geo_to_degrees(x), geo_to_degrees(y)) for x, y in coords]
    else:
        # For EUC_2D and CEIL_2D, no conversion needed
        return coords

def create_data_model(coords):
    """Create data model for the OR-Tools TSP solver."""
    data = {}
    data['distance_matrix'] = []
    for i in range(len(coords)):
        row = []
        for j in range(len(coords)):
            if i == j:
                row.append(0)
            else:
                row.append(int(distance(coords[i], coords[j])))
        data['distance_matrix'].append(row)
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

def distance(coord1, coord2):
    """Compute Euclidean distance between two points."""
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def print_solution(manager, routing, solution):
    """Print the solution in the desired format."""
    # Print the length of the shortest cycle
    print(solution.ObjectiveValue())

    # Print the tour
    index = routing.Start(0)
    tour = []
    while not routing.IsEnd(index):
        tour.append(manager.IndexToNode(index) + 1)  # Convert to 1-based index
        index = solution.Value(routing.NextVar(index))
    tour.append(manager.IndexToNode(index) + 1)  # Convert to 1-based index
    print(" ".join(map(str, tour)))

def solve_tsp(filename):
    # Parse the TSPLIB file and get coordinates and edge weight type
    edge_weight_type, coords = parse_tsp_file(filename)

    # Preprocess coordinates based on edge weight type
    coords = preprocess_coords(edge_weight_type, coords)

    # Create data model
    data = create_data_model(coords)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        # Returns the distance between the two nodes.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Set search parameters.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print the solution if available.
    if solution:
        print_solution(manager, routing, solution)
    else:
        print("No solution found.")

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python solve_tsp.py <input_file.tsp>")
        sys.exit(1)

    filename = sys.argv[1]
    solve_tsp(filename)

