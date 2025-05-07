import numpy as np

def shaw_removal(solution, data, seed=None, min_remove=2, max_remove=6):
    """
    Shaw removal operator: removes related calls based on distance, time, and load similarities.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert to list if numpy array
    if isinstance(solution, np.ndarray):
        solution = solution.tolist()
    
    # Find all calls in the solution (excluding 0s)
    calls = [x for x in solution if x > 0]
    unique_calls = list(set(calls))
    
    # Determine number of calls to remove
    n_to_remove = np.random.randint(min_remove, min(max_remove + 1, len(unique_calls) + 1))
    
    # Select a random call as seed
    seed_call = np.random.choice(unique_calls)
    calls_to_remove = [seed_call]
    remaining_calls = [c for c in unique_calls if c != seed_call]
    
    # Remove calls based on relatedness to seed call
    while len(calls_to_remove) < n_to_remove and remaining_calls:
        # Calculate relatedness scores for remaining calls
        relatedness_scores = []
        
        for call in remaining_calls:
            # Calculate relatedness based on multiple factors
            
            # 1. Load similarity (normalized difference in cargo size)
            load_diff = abs(data['Cargo'][seed_call-1, 2] - data['Cargo'][call-1, 2])
            max_load = max(data['Cargo'][:, 2]) if data['Cargo'].shape[0] > 0 else 1
            load_similarity = 1 - (load_diff / max_load)
            
            # 2. Safe distance calculation with fallback
            try:
                pickup_dist = calculate_node_distance(seed_call, call, data, is_pickup=True)
                delivery_dist = calculate_node_distance(seed_call, call, data, is_delivery=True)
            except (IndexError, TypeError):
                # Fallback to simple relatedness if distance calculation fails
                pickup_dist = 0.5
                delivery_dist = 0.5
            
            # 3. Vehicle compatibility similarity
            vehicle_compatibility = 0
            try:
                vehicle_compatibility = sum(1 for v in range(data['n_vehicles']) 
                                          if data['VesselCargo'][v][seed_call-1] == data['VesselCargo'][v][call-1])
                vehicle_similarity = vehicle_compatibility / data['n_vehicles']
            except (IndexError, TypeError):
                vehicle_similarity = 0.5
            
            # Weighted relatedness score (can be tuned)
            relatedness = 0.5 * (pickup_dist + delivery_dist) / 2 + 0.3 * load_similarity + 0.2 * vehicle_similarity
            relatedness_scores.append((call, relatedness))
        
        # Sort calls by relatedness (higher is more related)
        relatedness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Simple selection: pick the most related call
        next_call = relatedness_scores[0][0]
        
        # Add to removal list and remove from candidates
        calls_to_remove.append(next_call)
        remaining_calls.remove(next_call)
    
    # Remove the selected calls from the solution
    removed_solution = solution.copy()
    for call in calls_to_remove:
        # Find and remove both pickup and delivery
        pickup_idx = removed_solution.index(call)
        removed_solution.pop(pickup_idx)
        delivery_idx = removed_solution.index(call)
        removed_solution.pop(delivery_idx)
    
    return removed_solution, calls_to_remove

def shaw_removal_clipped(solution, data, seed=None, min_remove=6, max_remove=14):
    """
    Shaw removal operator: removes related calls based on distance, time, and load similarities.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert to list if numpy array
    if isinstance(solution, np.ndarray):
        solution = solution.tolist()
    
    # Find all calls in the solution (excluding 0s)
    calls = [x for x in solution if x > 0]
    unique_calls = list(set(calls))

    if len(unique_calls) < min_remove:
        min_remove = len(unique_calls)
    
    # Determine number of calls to remove
    n_to_remove = np.random.randint(min_remove, min(max_remove + 1, len(unique_calls) + 1))
    
    # Select a random call as seed
    seed_call = np.random.choice(unique_calls)
    calls_to_remove = [seed_call]
    remaining_calls = [c for c in unique_calls if c != seed_call]
    
    # Remove calls based on relatedness to seed call
    while len(calls_to_remove) < n_to_remove and remaining_calls:
        # Calculate relatedness scores for remaining calls
        relatedness_scores = []
        
        for call in remaining_calls:
            # Calculate relatedness based on multiple factors
            
            # 1. Load similarity (normalized difference in cargo size)
            load_diff = abs(data['Cargo'][seed_call-1, 2] - data['Cargo'][call-1, 2])
            max_load = max(data['Cargo'][:, 2]) if data['Cargo'].shape[0] > 0 else 1
            load_similarity = 1 - (load_diff / max_load)
            
            # 2. Safe distance calculation with fallback
            try:
                pickup_dist = calculate_node_distance(seed_call, call, data, is_pickup=True)
                delivery_dist = calculate_node_distance(seed_call, call, data, is_delivery=True)
            except (IndexError, TypeError):
                # Fallback to simple relatedness if distance calculation fails
                pickup_dist = 0.5
                delivery_dist = 0.5
            
            # 3. Vehicle compatibility similarity
            vehicle_compatibility = 0
            try:
                vehicle_compatibility = sum(1 for v in range(data['n_vehicles']) 
                                          if data['VesselCargo'][v][seed_call-1] == data['VesselCargo'][v][call-1])
                vehicle_similarity = vehicle_compatibility / data['n_vehicles']
            except (IndexError, TypeError):
                vehicle_similarity = 0.5
            
            # Weighted relatedness score (can be tuned)
            relatedness = 0.5 * (pickup_dist + delivery_dist) / 2 + 0.3 * load_similarity + 0.2 * vehicle_similarity
            relatedness_scores.append((call, relatedness))
        
        # Sort calls by relatedness (higher is more related)
        relatedness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Simple selection: pick the most related call
        next_call = relatedness_scores[0][0]
        
        # Add to removal list and remove from candidates
        calls_to_remove.append(next_call)
        remaining_calls.remove(next_call)
    
    # Remove the selected calls from the solution
    removed_solution = solution.copy()
    for call in calls_to_remove:
        # Find and remove both pickup and delivery
        pickup_idx = removed_solution.index(call)
        removed_solution.pop(pickup_idx)
        delivery_idx = removed_solution.index(call)
        removed_solution.pop(delivery_idx)
    
    return removed_solution, calls_to_remove


def calculate_node_distance(call1, call2, data, is_pickup=False, is_delivery=False):
    """
    Calculate normalized distance between nodes of two calls.
    """
    try:
        # Get node indices and ensure they're integers
        if is_pickup:
            node1 = int(data['Cargo'][call1-1, 0])
            node2 = int(data['Cargo'][call2-1, 0])
        elif is_delivery:
            node1 = int(data['Cargo'][call1-1, 1])
            node2 = int(data['Cargo'][call2-1, 1])
        else:
            # Average of pickup and delivery distances
            return (calculate_node_distance(call1, call2, data, is_pickup=True) + 
                    calculate_node_distance(call1, call2, data, is_delivery=True)) / 2
        
        # Safely access travel time data
        travel_times = data['TravelTime']
        if isinstance(travel_times, np.ndarray) and travel_times.ndim == 3:
            distance = travel_times[0, node1, node2]  # Using comma notation
        elif isinstance(travel_times, list):
            distance = travel_times[0][node1][node2]  # Using bracket notation
        else:
            # Fallback if structure is different
            return 0.5  # Default middle value
        
        # Normalize by maximum distance
        max_dist = 1.0  # Default
        try:
            if isinstance(travel_times, np.ndarray):
                max_dist = np.max(travel_times[0]) if travel_times[0].size > 0 else 1.0
            else:
                max_dist = max(max(row) for row in travel_times[0]) if travel_times[0] else 1.0
        except:
            pass
        
        normalized_dist = 1 - (distance / max_dist)  # Higher value means closer nodes
        return normalized_dist
    
    except (IndexError, TypeError, ValueError) as e:
        # Fallback to a neutral value if any error occurs
        return 0.5