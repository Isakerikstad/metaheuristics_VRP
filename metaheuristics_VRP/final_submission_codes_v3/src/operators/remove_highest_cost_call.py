import numpy as np
from src.utils.Utils import cost_function

def remove_highest_cost_call(solution, data, seed=None, min_remove=2, max_remove=10):
    """
    Removes 'n' calls from the solution, with 'n' in [min_remove, max_remove].
    Calls with higher added cost to the solution have higher probability of being removed.
    The delta_cost weighting stands for 70% of the chance of removal while the remaining 30% is random.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert to list if numpy array
    if isinstance(solution, np.ndarray):
        solution = solution.tolist()
    
    # Find vehicle separators
    separator_indices = [i for i, x in enumerate(solution) if x == 0]
    n_vehicles = len(separator_indices)
    
    # Gather all unique calls (excluding zeros)
    unique_calls = {}  # Dictionary to store {call_id: (vehicle_idx, positions)}
    
    for v in range(n_vehicles):
        start_idx = 0 if v == 0 else separator_indices[v-1] + 1
        end_idx = separator_indices[v]
        route = solution[start_idx:end_idx]
        
        # Track pickup and delivery positions for each call in this route
        for i, call in enumerate(route):
            if call > 0:
                if call not in unique_calls:
                    unique_calls[call] = (v, [i + start_idx])
                else:
                    # Add second position (delivery)
                    unique_calls[call] = (v, unique_calls[call][1] + [i + start_idx])
    
    # Filter calls that are already in the solution with both pickup and delivery
    complete_calls = {call: data for call, data in unique_calls.items() 
                     if len(data[1]) == 2}
    
    if not complete_calls:
        return np.array(solution), []
    
    # Calculate delta cost (cost savings) for each call
    call_costs = []
    
    for call, (vehicle_idx, positions) in complete_calls.items():
        # Calculate cost with the call
        current_cost = cost_function(solution, data)
        
        # Calculate cost without the call
        temp_solution = solution.copy()
        # Remove in reverse order to keep indices valid
        temp_solution.pop(positions[1])
        temp_solution.pop(positions[0])
        
        # Calculate new cost after removal
        new_cost = cost_function(temp_solution, data)
        
        # Delta cost is the savings from removing this call
        delta_cost = current_cost - new_cost
        
        # Store call and its delta cost
        call_costs.append((call, delta_cost))
    
    # Decide how many calls to remove
    n_to_remove = np.random.randint(min(min_remove, len(complete_calls)), min(max_remove, len(complete_calls)) + 1)
    
    # Sort calls by delta cost (highest first)
    call_costs.sort(key=lambda x: x[1], reverse=True)
    
    # Select calls to remove with probability biased toward higher delta cost
    calls_to_remove = []
    remaining_calls = call_costs.copy()
    
    for _ in range(n_to_remove):
        if not remaining_calls:
            break
        
        # 100% chance of selecting based on delta cost, 0% random
        if np.random.random() < 1:
            # Select based on cost (prefer higher delta cost)
            weights = np.array([c[1] for c in remaining_calls])
            # Handle negative weights by shifting to positive
            if np.min(weights) < 0:
                weights = weights - np.min(weights) + 1e-6
            
            # Compute probabilities
            probs = weights / weights.sum() if weights.sum() > 0 else None
            
            # Select call with probability proportional to cost savings
            selected_idx = np.random.choice(len(remaining_calls), p=probs)
        else:
            # Select randomly
            selected_idx = np.random.randint(0, len(remaining_calls))
        
        calls_to_remove.append(remaining_calls[selected_idx][0])
        remaining_calls.pop(selected_idx)
    
    # Remove the chosen calls from solution
    new_solution = solution.copy()
    
    # Store positions of all calls to remove
    positions_to_remove = []
    for call in calls_to_remove:
        vehicle_idx, call_positions = complete_calls[call]
        positions_to_remove.extend(call_positions)
    
    # Sort positions in descending order to remove from end to beginning
    positions_to_remove.sort(reverse=True)
    
    # Remove calls at specified positions
    for pos in positions_to_remove:
        new_solution.pop(pos)

    # shuffle the removed calls
    np.random.shuffle(calls_to_remove)
    
    return new_solution, calls_to_remove

def remove_highest_cost_call_clipped(solution, data, seed=None, min_remove=6, max_remove=8):
    """
    Removes 'n' calls from the solution, with 'n' in [min_remove, max_remove].
    Calls with higher added cost to the solution have higher probability of being removed.
    The delta_cost weighting stands for 70% of the chance of removal while the remaining 30% is random.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert to list if numpy array
    if isinstance(solution, np.ndarray):
        solution = solution.tolist()
    
    # Find vehicle separators
    separator_indices = [i for i, x in enumerate(solution) if x == 0]
    n_vehicles = len(separator_indices)
    
    # Gather all unique calls (excluding zeros)
    unique_calls = {}  # Dictionary to store {call_id: (vehicle_idx, positions)}
    
    for v in range(n_vehicles):
        start_idx = 0 if v == 0 else separator_indices[v-1] + 1
        end_idx = separator_indices[v]
        route = solution[start_idx:end_idx]
        
        # Track pickup and delivery positions for each call in this route
        for i, call in enumerate(route):
            if call > 0:
                if call not in unique_calls:
                    unique_calls[call] = (v, [i + start_idx])
                else:
                    # Add second position (delivery)
                    unique_calls[call] = (v, unique_calls[call][1] + [i + start_idx])
    
    # Filter calls that are already in the solution with both pickup and delivery
    complete_calls = {call: data for call, data in unique_calls.items() 
                     if len(data[1]) == 2}
    
    if not complete_calls:
        return np.array(solution), []
    
    # Calculate delta cost (cost savings) for each call
    call_costs = []
    
    for call, (vehicle_idx, positions) in complete_calls.items():
        # Calculate cost with the call
        current_cost = cost_function(solution, data)
        
        # Calculate cost without the call
        temp_solution = solution.copy()
        # Remove in reverse order to keep indices valid
        temp_solution.pop(positions[1])
        temp_solution.pop(positions[0])
        
        # Calculate new cost after removal
        new_cost = cost_function(temp_solution, data)
        
        # Delta cost is the savings from removing this call
        delta_cost = current_cost - new_cost
        
        # Store call and its delta cost
        call_costs.append((call, delta_cost))
    
    # Decide how many calls to remove
    n_to_remove = np.random.randint(min(min_remove, len(complete_calls)), min(max_remove, len(complete_calls)) + 1)
    
    # Sort calls by delta cost (highest first)
    call_costs.sort(key=lambda x: x[1], reverse=True)
    
    # Select calls to remove with probability biased toward higher delta cost
    calls_to_remove = []
    remaining_calls = call_costs.copy()
    
    for _ in range(n_to_remove):
        if not remaining_calls:
            break
        
        # 100% chance of selecting based on delta cost, 0% random
        if np.random.random() < 1:
            # Select based on cost (prefer higher delta cost)
            weights = np.array([c[1] for c in remaining_calls])
            # Handle negative weights by shifting to positive
            if np.min(weights) < 0:
                weights = weights - np.min(weights) + 1e-6
            
            # Compute probabilities
            probs = weights / weights.sum() if weights.sum() > 0 else None
            
            # Select call with probability proportional to cost savings
            selected_idx = np.random.choice(len(remaining_calls), p=probs)
        else:
            # Select randomly
            selected_idx = np.random.randint(0, len(remaining_calls))
        
        calls_to_remove.append(remaining_calls[selected_idx][0])
        remaining_calls.pop(selected_idx)
    
    # Remove the chosen calls from solution
    new_solution = solution.copy()
    
    # Store positions of all calls to remove
    positions_to_remove = []
    for call in calls_to_remove:
        vehicle_idx, call_positions = complete_calls[call]
        positions_to_remove.extend(call_positions)
    
    # Sort positions in descending order to remove from end to beginning
    positions_to_remove.sort(reverse=True)
    
    # Remove calls at specified positions
    for pos in positions_to_remove:
        new_solution.pop(pos)

    # shuffle the removed calls
    np.random.shuffle(calls_to_remove)
    
    return new_solution, calls_to_remove

# Alternative implementation with advanced delta cost calculation
def calculate_call_delta_cost(solution, data, call):
    """
    Calculate the delta cost (marginal cost) of a call in the solution.
    
    Args:
        solution: Current solution
        data: Problem data
        call: Call ID to evaluate
        
    Returns:
        float: Delta cost (positive means removing saves cost)
    """
    # Find positions of this call in the solution
    positions = [i for i, x in enumerate(solution) if x == call]
    if len(positions) != 2:
        return 0  # Call not fully in solution
        
    # Find which vehicle contains this call
    separator_indices = [i for i, x in enumerate(solution) if x == 0]
    vehicle_idx = next((i for i, sep in enumerate(separator_indices) if sep > positions[0]), 0)
    
    # Extract the current route for this vehicle
    start_idx = 0 if vehicle_idx == 0 else separator_indices[vehicle_idx-1] + 1
    end_idx = separator_indices[vehicle_idx]
    current_route = solution[start_idx:end_idx]
    
    # Calculate cost with the call
    current_cost = cost_function(solution, data)
    
    # Remove the call
    temp_solution = solution.copy()
    # Remove in reverse order to keep indices valid
    temp_solution.pop(positions[1])
    temp_solution.pop(positions[0])
    
    # Calculate cost without the call
    new_cost = cost_function(temp_solution, data)
    
    # Return delta cost (positive means removing saves cost)
    return current_cost - new_cost