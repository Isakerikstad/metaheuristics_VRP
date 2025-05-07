import numpy as np

def dummy_removal(solution, data, seed=None, min_remove=3, max_remove=8):
    """
    Removal operator that specifically targets calls assigned to the dummy vehicle (outsourced calls).
    Prioritizes removing calls that are expensive to outsource and compatible with real vehicles.
    
    Args:
        solution: Current solution
        data: Problem data dictionary
        seed: Random seed for reproducibility
        min_remove: Minimum number of calls to remove
        max_remove: Maximum number of calls to remove
        
    Returns:
        Tuple of (modified solution, list of removed calls)
    """
    if seed is not None:
        np.random.seed(seed)

    n_vehicles = data['n_vehicles']
    
    # Convert to list if numpy array
    if isinstance(solution, np.ndarray):
        solution = solution.tolist()
    
    # Find the start index of dummy vehicle. last zero solution
    dummy_start_idx = len(solution) - 1 - solution[::-1].index(0)
    
    # Get calls in dummy vehicle
    dummy_route = solution[dummy_start_idx:]
    dummy_calls = [c for c in dummy_route if c > 0]
    unique_dummy_calls = list(set(dummy_calls))
    
    if not unique_dummy_calls:
        # No calls in dummy vehicle, return original solution
        return np.array(solution), []
    
    # Score calls based on outsourcing cost and compatibility with real vehicles
    call_scores = []
    
    for call in unique_dummy_calls:
        try:
            # Get outsource cost (higher cost = higher priority to remove)
            outsource_cost = data['Cargo'][call - 1, 3]
            
            # Check compatibility with real vehicles
            compatible_vehicles = 0
            for v in range(n_vehicles - 1):  # Exclude dummy vehicle
                if data['VesselCargo'][v][call - 1] == 1:
                    compatible_vehicles += 1
            
            compatibility_ratio = compatible_vehicles / (n_vehicles - 1) if n_vehicles > 1 else 0
            
            # Calculate score (higher is better candidate for removal) + some a gaussian noise centered around 0 and sigma = 0.3
            score = outsource_cost * (0.5 + 0.5 * compatibility_ratio + np.random.normal(0, 0.3))
            call_scores.append((call, score))
            
            
        except (IndexError, KeyError):
            # If we can't calculate score, use default
            call_scores.append((call, 1.0))
    
    # Sort by score (highest first)
    call_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Determine number of calls to remove
    num_to_remove = min(np.random.randint(min_remove, max_remove + 1), len(call_scores))
    
    # Select top scoring calls
    calls_to_remove = [call for call, _ in call_scores[:num_to_remove]]
    
    # Remove the calls
    removed_solution = solution.copy()
    
    # Remove from end to start to avoid index issues
    for call in calls_to_remove:
        # Find positions of the call in the solution
        positions = [i for i, x in enumerate(removed_solution) if x == call]
        if len(positions) >= 2:
            # Remove in reverse order (delivery first, then pickup)
            removed_solution.pop(positions[1])
            removed_solution.pop(positions[0])
    
    return removed_solution, calls_to_remove

def dummy_removal_clipped(solution, data, seed=None, min_remove=6, max_remove=8):
    """
    Removal operator that specifically targets calls assigned to the dummy vehicle (outsourced calls).
    Prioritizes removing calls that are expensive to outsource and compatible with real vehicles.
    
    Args:
        solution: Current solution
        data: Problem data dictionary
        seed: Random seed for reproducibility
        min_remove: Minimum number of calls to remove
        max_remove: Maximum number of calls to remove
        
    Returns:
        Tuple of (modified solution, list of removed calls)
    """
    if seed is not None:
        np.random.seed(seed)

    n_vehicles = data['n_vehicles']
    
    # Convert to list if numpy array
    if isinstance(solution, np.ndarray):
        solution = solution.tolist()
    
    # Find the start index of dummy vehicle. last zero solution
    dummy_start_idx = len(solution) - 1 - solution[::-1].index(0)
    
    # Get calls in dummy vehicle
    dummy_route = solution[dummy_start_idx:]
    dummy_calls = [c for c in dummy_route if c > 0]
    unique_dummy_calls = list(set(dummy_calls))
    
    if not unique_dummy_calls:
        # No calls in dummy vehicle, return original solution
        return np.array(solution), []
    
    # Score calls based on outsourcing cost and compatibility with real vehicles
    call_scores = []
    
    for call in unique_dummy_calls:
        try:
            # Get outsource cost (higher cost = higher priority to remove)
            outsource_cost = data['Cargo'][call - 1, 3]
            
            # Check compatibility with real vehicles
            compatible_vehicles = 0
            for v in range(n_vehicles - 1):  # Exclude dummy vehicle
                if data['VesselCargo'][v][call - 1] == 1:
                    compatible_vehicles += 1
            
            compatibility_ratio = compatible_vehicles / (n_vehicles - 1) if n_vehicles > 1 else 0
            
            # Calculate score (higher is better candidate for removal) + some a gaussian noise centered around 0 and sigma = 0.3
            score = outsource_cost * (0.5 + 0.5 * compatibility_ratio + np.random.normal(0, 0.3))
            call_scores.append((call, score))
            
            
        except (IndexError, KeyError):
            # If we can't calculate score, use default
            call_scores.append((call, 1.0))
    
    # Sort by score (highest first)
    call_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Determine number of calls to remove
    num_to_remove = min(np.random.randint(min_remove, max_remove + 1), len(call_scores))
    
    # Select top scoring calls
    calls_to_remove = [call for call, _ in call_scores[:num_to_remove]]
    
    # Remove the calls
    removed_solution = solution.copy()
    
    # Remove from end to start to avoid index issues
    for call in calls_to_remove:
        # Find positions of the call in the solution
        positions = [i for i, x in enumerate(removed_solution) if x == call]
        if len(positions) >= 2:
            # Remove in reverse order (delivery first, then pickup)
            removed_solution.pop(positions[1])
            removed_solution.pop(positions[0])
    
    return removed_solution, calls_to_remove