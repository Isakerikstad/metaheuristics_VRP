import numpy as np

from src.utils.Utils import feasibility_check, cost_function

def insert_legally(solution, calls_to_move, data, seed=None):
    """
    Insert calls into real vehicles while ensuring that the solution remains feasible.
    """

    if seed is not None:
        np.random.seed(seed)

    for a in range(len(calls_to_move)):           

        # available_vehicles in numpy array
        available_vehicles = np.array(range(data['n_vehicles']))        

        # Remove all vehicles with uncomapatible with our call and capacity
        available_vehicles = np.array([v for v in available_vehicles if data['VesselCargo'][v][calls_to_move[a] - 1] == 1])

        target_vehicle = np.random.choice(available_vehicles)

        # Find vehicle boundaries (positions of 0s)
        vehicle_bounds = [i for i, x in enumerate(solution) if x == 0]
        
        # Get the range for the target vehicle
        start_idx = 0 if target_vehicle == 0 else vehicle_bounds[target_vehicle - 1] + 1
        end_idx = vehicle_bounds[target_vehicle] if target_vehicle < len(vehicle_bounds) else len(solution)

        # Choose random positions for reinsertion within the chosen vehicle's route
        route_length = end_idx - start_idx
        if route_length >= 2:
            # Generate two random positions within the vehicle's route
            pos1 = np.random.randint(start_idx, end_idx + 1)
            pos2 = np.random.randint(start_idx, end_idx + 2)
            
            # Insert the call at the chosen positions
            solution.insert(pos1, calls_to_move[a])
            solution.insert(pos2, calls_to_move[a])
        else:
            # If not enough space, insert at the start of the route
            solution.insert(start_idx, calls_to_move[a])
            solution.insert(start_idx, calls_to_move[a])

    return np.array(solution)
