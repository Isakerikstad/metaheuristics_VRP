"""Swap vehicles' calls if feasible and if it improves the solution cost."""
import numpy as np
from src.utils.Utils import vehicle_feasibility_check, cost_function

def vehicle_swap(solution, data, seed=None, num_vehicles_to_swap=2):
    """
    Swap calls between vehicles in a cyclic manner if feasible and if it improves the solution cost.
    
    Args:
        solution: Current solution array
        data: Problem data dictionary
        seed: Random seed for randomization
        num_vehicles_to_swap: Number of vehicles to include in the cyclic swap (default: 2)
        
    Returns:
        solution: Updated solution after beneficial swaps
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert to list for easier manipulation
    solution_list = solution.tolist() if isinstance(solution, np.ndarray) else solution.copy()
    n_vehicles = data['n_vehicles']
    
    # Calculate initial solution cost
    initial_cost = cost_function(solution_list, data)
    best_solution = solution_list.copy()
    
    # Skip if there aren't enough vehicles to swap
    if num_vehicles_to_swap > n_vehicles - 1:  # Exclude dummy vehicle
        return np.array(solution_list)
    
    # Create list of real vehicles (exclude dummy vehicle)
    real_vehicles = list(range(n_vehicles - 1))
    
    # Find indices of vehicle separators (zeros)
    vehicle_separators = []
    for i, val in enumerate(solution_list):
        if val == 0:
            vehicle_separators.append(i)
    
    # Maps vehicle index to (start_idx, end_idx) in solution
    vehicle_sections = {}
    for v in range(n_vehicles):
        # For first vehicle
        if v == 0:
            start_idx = 0
        else:
            start_idx = vehicle_separators[v-1] + 1
        
        # For last (dummy) vehicle
        if v == n_vehicles - 1:
            end_idx = len(solution_list)
        else:
            end_idx = vehicle_separators[v]
        
        vehicle_sections[v] = (start_idx, end_idx)
    
    # Extract calls for each vehicle
    vehicle_calls = {}
    for v, (start_idx, end_idx) in vehicle_sections.items():
        vehicle_calls[v] = solution_list[start_idx:end_idx]
    
    # Create set of unique calls per vehicle for compatibility checking
    unique_calls_per_vehicle = {}
    for v, calls in vehicle_calls.items():
        unique_calls_per_vehicle[v] = set([c for c in calls if c > 0])
    
    improvements_found = 0
    swap_attempts = 0
    max_attempts = min(20, n_vehicles * (n_vehicles - 1) // 2)
    
    while swap_attempts < max_attempts and len(real_vehicles) >= num_vehicles_to_swap:
        swap_attempts += 1
        
        try:
            # Select random subset of vehicles to swap (only from real vehicles, not dummy)
            if len(real_vehicles) > num_vehicles_to_swap:
                vehicles_to_swap = sorted(np.random.choice(real_vehicles, size=num_vehicles_to_swap, replace=False))
            else:
                vehicles_to_swap = real_vehicles.copy()
            
            # Create a circular sequence for swapping (v1 -> v2 -> v3 -> v1)
            swap_sequence = vehicles_to_swap + [vehicles_to_swap[0]]
            
            # Check cargo compatibility
            compatible = True
            for i in range(num_vehicles_to_swap):
                source_vehicle = swap_sequence[i]
                target_vehicle = swap_sequence[i+1]
                
                for call in unique_calls_per_vehicle[source_vehicle]:
                    if data['VesselCargo'][target_vehicle][call - 1] == 0:
                        compatible = False
                        break
                
                if not compatible:
                    break
            
            if not compatible:
                continue
            
            # Create a new solution by combining sections
            new_solution = []
            
            # Create a mapping for the swapped vehicle calls
            swapped_calls = {}
            for i in range(num_vehicles_to_swap):
                source_vehicle = swap_sequence[i]
                target_vehicle = swap_sequence[i+1]
                swapped_calls[target_vehicle] = vehicle_calls[source_vehicle]
            
            # Build the new solution
            for v in range(n_vehicles):
                # Add calls for this vehicle
                if v in swapped_calls:
                    new_solution.extend(swapped_calls[v])  # Use swapped calls
                else:
                    new_solution.extend(vehicle_calls[v])  # Keep original calls
                
                # Add vehicle separator except after last vehicle
                if v < n_vehicles - 1:
                    new_solution.append(0)
            
            # Check feasibility for each modified vehicle
            all_feasible = True
            for v in vehicles_to_swap:
                feasible, _ = vehicle_feasibility_check(new_solution, data, v)
                if not feasible:
                    all_feasible = False
                    break
                    
            if all_feasible:
                # Calculate cost of new solution
                new_cost = cost_function(new_solution, data)
                
                # Keep the swap if it improves the solution
                if new_cost < initial_cost:
                    best_solution = new_solution.copy()
                    initial_cost = new_cost
                    improvements_found += 1
                    #print(f"Improved solution with {num_vehicles_to_swap}-way swap of vehicles {vehicles_to_swap}")
        
        except Exception as e:
            print(f"Error during vehicle swap: {e}")
        
    return np.array(best_solution)