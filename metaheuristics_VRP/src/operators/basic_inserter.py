import numpy as np
    
def basic_inserter(solution, calls_to_move, data, seed=None):
    """
    Local search using 1-reinsert operator: removes one call and reinserts it in a new position.
    Handles constraints and ensures meaningful neighborhood moves.
    """
    if seed is not None:
        np.random.seed(seed)
     
    n_vehicles = data['n_vehicles']

    for a in range(len(calls_to_move)):           

        # available_vehicles in numpy array
        available_vehicles = np.array(range(n_vehicles))        

        # Remove all vehicles with uncomapatible with our call and capacity
        available_vehicles = np.array([v for v in available_vehicles if data['VesselCargo'][v][calls_to_move[a] - 1] == 1])
        
        # Check call size and call outsource cost against mean
        large_call = False
        call_size = data['Cargo'][calls_to_move[a] - 1, 2]
        if call_size > data['average_size']:
            large_call = True

        expensive_call = False
        call_cost = data['Cargo'][calls_to_move[a] - 1, 3]
        if call_cost > data['average_cost_outsource']:
            expensive_call = True

        # Over available vehicles, give property True if vehicle capacity over mean
        probs = np.ones(len(available_vehicles))
        large_capacity = np.array([data['VesselCapacity'][v] > data['average_capacity'] for v in available_vehicles])

        if large_call:
            # Larger probability for vehicles above average capacity
            probs = np.where(large_capacity, 0.9, 0.6)
            probs[-1] = 1

        else:
            # Larger probability for vehicles below average capacity
            probs = np.where(large_capacity, 0.8, 1)

        if expensive_call:
            # lower probability for dummy
            probs[-1] = 0.8

        # Normalize probabilities
        probs = probs / np.sum(probs)
        #print(available_vehicles, probs)

        # Use the computed distribution to choose vehicles
        available_vehicles = np.random.choice(available_vehicles, size=len(available_vehicles), replace=False, p=probs)
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

def basic_inserter_v2(solution, calls_to_move, data, seed=None):
    """
    Variation 2: Load-aware insertion that considers current vehicle loads.
    """
    if seed is not None:
        np.random.seed(seed)
     
    n_vehicles = data['n_vehicles']


    # Calculate current load of each vehicle
    vehicle_loads = [0] * n_vehicles
    vehicle_bounds = [i for i, x in enumerate(solution) if x == 0]
    
    for v in range(n_vehicles):
        start_idx = 0 if v == 0 else vehicle_bounds[v - 1] + 1
        end_idx = vehicle_bounds[v] if v < len(vehicle_bounds) else len(solution)
        
        route = solution[start_idx:end_idx]
        calls_in_route = set()
        
        for call in route:
            if call > 0:
                if call in calls_in_route:
                    calls_in_route.remove(call)  # Delivery
                else:
                    calls_in_route.add(call)     # Pickup
                    vehicle_loads[v] += data['Cargo'][call - 1, 2]

    # Process each call
    for a in range(len(calls_to_move)):           
        # Find compatible vehicles
        available_vehicles = np.array(range(n_vehicles))        
        available_vehicles = np.array([v for v in available_vehicles if data['VesselCargo'][v][calls_to_move[a] - 1] == 1])
        
        # Check call size and cost
        call_size = data['Cargo'][calls_to_move[a] - 1, 2]
        call_cost = data['Cargo'][calls_to_move[a] - 1, 3]
        large_call = call_size > data['average_size']

        # Calculate load percentages for available vehicles
        load_percentages = []
        for v in available_vehicles:
            if v < n_vehicles - 1:  # Not dummy
                capacity = data['VesselCapacity'][v]
                current_load = vehicle_loads[v]
                percentage = current_load / capacity if capacity > 0 else 1.0
                load_percentages.append(percentage)
            else:
                load_percentages.append(0)  # Dummy has no load
                
        load_percentages = np.array(load_percentages)
        
        # Assign probabilities based on available space
        probs = 1.0 - load_percentages  # More space = higher probability
        probs = np.clip(probs, 0.1, 1.0)  # Ensure minimum probability
        
        if large_call:
            # Emphasize vehicles with more free space
            probs = probs ** 1.5
        
        # Special handling for dummy vehicle
        if available_vehicles[-1] == n_vehicles - 1:
            if call_cost > data['average_cost_outsource']:
                # Expensive call - lower dummy probability
                probs[-1] = 0.3
            else:
                # Not expensive - higher dummy probability
                probs[-1] = 0.7
        
        # Normalize probabilities
        probs = probs / np.sum(probs)

        # Select target vehicle
        target_vehicle = np.random.choice(available_vehicles, p=probs)

        # Update vehicle bounds for insertion
        vehicle_bounds = [i for i, x in enumerate(solution) if x == 0]
        start_idx = 0 if target_vehicle == 0 else vehicle_bounds[target_vehicle - 1] + 1
        end_idx = vehicle_bounds[target_vehicle] if target_vehicle < len(vehicle_bounds) else len(solution)
        
        # Insert call
        route_length = end_idx - start_idx
        if route_length >= 2:
            pos1 = np.random.randint(start_idx, end_idx + 1)
            pos2 = np.random.randint(pos1, end_idx + 2) if pos1 < end_idx + 1 else pos1 + 1
            
            solution.insert(pos1, calls_to_move[a])
            solution.insert(pos2, calls_to_move[a])
            
            # Update vehicle load
            if target_vehicle < n_vehicles - 1:
                vehicle_loads[target_vehicle] += call_size
        else:
            # Insert at beginning
            solution.insert(start_idx, calls_to_move[a])
            solution.insert(start_idx, calls_to_move[a])
            
            # Update vehicle load
            if target_vehicle < n_vehicles - 1:
                vehicle_loads[target_vehicle] += call_size
    
    return np.array(solution)
    
def basic_inserter_v4blabla(solution, calls_to_move, data, seed=None):
    """
    Variation 4: Uses a demand-balanced strategy to distribute calls
    more evenly among vehicles.
    """
    if seed is not None:
        np.random.seed(seed)

    if isinstance(solution, np.ndarray):
        solution = solution.tolist()
     
    n_vehicles = data['n_vehicles']
    print(calls_to_move)
    # Small chance of inserting all to dummy (4%)
    dummy_pos = len(solution) - 1
    if np.random.rand() < 0.04:
        for call in calls_to_move:
            solution.insert(dummy_pos, call)
            solution.insert(dummy_pos + 1, call)
            dummy_pos += 2
    
        return np.array(solution)
    
    # Calculate current distribution of calls per vehicle
    vehicle_counts = [0] * n_vehicles
    vehicle_bounds = [i for i, x in enumerate(solution) if x == 0]
    
    for v in range(n_vehicles - 1):  # Exclude dummy
        start_idx = 0 if v == 0 else vehicle_bounds[v - 1] + 1
        end_idx = vehicle_bounds[v] if v < len(vehicle_bounds) else len(solution)
        
        # Count unique calls 
        unique_calls = set(solution[start_idx:end_idx])
        if 0 in unique_calls:
            unique_calls.remove(0)
        vehicle_counts[v] = len(unique_calls)
    
    # Average calls per real vehicle
    avg_calls = sum(vehicle_counts) / max(1, sum(1 for c in vehicle_counts if c > 0))
    
    # Process calls to insert
    random_order = np.random.permutation(len(calls_to_move))
    
    for a in random_order:           
        call = calls_to_move[a]
        
        # Find compatible vehicles
        available_vehicles = np.array([v for v in range(n_vehicles) if data['VesselCargo'][v][call - 1] == 1])
        
        # Call characteristics
        call_size = data['Cargo'][call - 1, 2]
        call_cost = data['Cargo'][call - 1, 3]
        
        # Calculate demand balancing penalties - prefer under-utilized vehicles
        balance_factors = np.ones(len(available_vehicles))
        
        for i, v in enumerate(available_vehicles):
            if v < n_vehicles - 1:  # Real vehicle
                # Calculate how full this vehicle is compared to average
                fullness_ratio = vehicle_counts[v] / max(1, avg_calls)
                
                # Penalty increases as vehicle gets more calls than average
                if fullness_ratio > 1:
                    balance_factors[i] = 1 / fullness_ratio
        
        # Base probabilities
        probs = np.ones(len(available_vehicles))
        
        # Adjust for large calls
        if call_size > data['average_size']:
            for i, v in enumerate(available_vehicles):
                if v < n_vehicles - 1:  # Real vehicle
                    if data['VesselCapacity'][v] < call_size * 2:
                        probs[i] *= 0.7  # Reduce probability for vehicles that would be too full
        
        # Apply demand balancing
        probs = probs * balance_factors
        
        # Special handling for dummy vehicle
        if available_vehicles[-1] == n_vehicles - 1:
            outsource_score = call_cost / data['average_cost_outsource']
            if outsource_score < 0.8:
                # Inexpensive to outsource - increase dummy probability
                probs[-1] = 1.5
            else:
                # Expensive to outsource - reduce dummy probability
                probs[-1] = 0.7
        
        # Normalize probabilities
        probs = probs / np.sum(probs)

        # Choose target vehicle
        target_vehicle = np.random.choice(available_vehicles, p=probs)

        # Update vehicle boundaries
        vehicle_bounds = [i for i, x in enumerate(solution) if x == 0]
        start_idx = 0 if target_vehicle == 0 else vehicle_bounds[target_vehicle - 1] + 1
        end_idx = vehicle_bounds[target_vehicle] if target_vehicle < len(vehicle_bounds) else len(solution)
        
        # Choose insertion positions
        if target_vehicle == n_vehicles - 1:
            # Dummy vehicle - just append
            solution.append(call)
            solution.append(call)
        else:
            route_length = end_idx - start_idx
            if route_length >= 2:
                # For regular vehicles, choose random positions
                # 30% chance to use sequential insertion
                if np.random.rand() < 0.3:
                    # Sequential insertion (pickup-delivery close together)
                    pos1 = np.random.randint(start_idx, end_idx + 1)
                    solution.insert(pos1, call)
                    solution.insert(pos1 + 1, call)
                else:
                    # Random positions
                    pos1 = np.random.randint(start_idx, end_idx + 1)
                    pos2 = np.random.randint(start_idx, end_idx + 2)
                    
                    # Ensure delivery is after pickup
                    if pos2 < pos1:
                        pos1, pos2 = pos2, pos1
                        
                    solution.insert(pos1, call)
                    solution.insert(pos2, call)
            else:
                # Empty route - insert at start
                solution.insert(start_idx, call)
                solution.insert(start_idx, call)
                
        # Update vehicle counts
        if target_vehicle < n_vehicles - 1:
            vehicle_counts[target_vehicle] += 1
    
    return np.array(solution)

def basic_inserter_v4(solution, calls_to_move, data, seed=None):
    """
    Enhanced clustering-based inserter that:
    1. Uses proximity information to group calls from the same cluster
    2. Prioritizes keeping related calls in the same vehicle
    3. Uses travel cost information to optimize insertion positions
    4. Fixes NaN probabilities issue in v4 version
    
    Uses the clustering data calculated by calculate_clustering() in Utils.py.
    """
    if seed is not None:
        np.random.seed(seed)

    if isinstance(solution, np.ndarray):
        solution = solution.tolist()
     
    n_vehicles = data['n_vehicles']
    
    # Ensure we're working with call indices (not call-index arrays)
    if isinstance(calls_to_move[0], (list, np.ndarray)):
        calls_to_move = [c for sublist in calls_to_move for c in sublist]

    # Calculate current distribution of calls per vehicle
    vehicle_calls = [set() for _ in range(n_vehicles)]
    vehicle_bounds = [i for i, x in enumerate(solution) if x == 0]
    
    # Get calls currently in each vehicle
    for v in range(n_vehicles):
        start_idx = 0 if v == 0 else vehicle_bounds[v - 1] + 1
        end_idx = vehicle_bounds[v] if v < len(vehicle_bounds) else len(solution)
        
        # Extract unique calls in this vehicle
        for call in solution[start_idx:end_idx]:
            if call > 0:
                vehicle_calls[v].add(call)
    
    # Group calls to move by their clusters
    cluster_grouped_calls = {}
    for call in calls_to_move:
        cluster_id = data['cluster_assignments'][call - 1]
        if cluster_id not in cluster_grouped_calls:
            cluster_grouped_calls[cluster_id] = []
        cluster_grouped_calls[cluster_id].append(call)
    
    # Randomize the order of cluster processing
    cluster_keys = list(cluster_grouped_calls.keys())
    np.random.shuffle(cluster_keys)
    
    # Process each cluster group
    for cluster_id in cluster_keys:
        calls_in_cluster = cluster_grouped_calls[cluster_id]
        
        # Sort calls within cluster by proximity to optimize sequence
        # We'll use a simple heuristic - sort by pickup node ID as a proxy for proximity
        calls_in_cluster.sort(key=lambda c: int(data['Cargo'][c - 1, 0]))
        
        # Group similar calls (up to 4 per vehicle) to keep related calls together
        max_calls_together = min(4, len(calls_in_cluster))
        calls_subgroups = []
        
        # Create subgroups with related calls
        for i in range(0, len(calls_in_cluster), max_calls_together):
            subgroup_size = min(max_calls_together, len(calls_in_cluster) - i)
            calls_subgroups.append(calls_in_cluster[i:i+subgroup_size])
        
        # Process each subgroup
        for subgroup in calls_subgroups:
            # Choose best vehicle based on cluster relationships
            best_vehicle = select_best_vehicle_v5(solution, subgroup, data, vehicle_calls)
            
            # Now insert calls in an optimized order based on proximity
            for call in subgroup:
                # Find optimal insertion positions
                pickup_pos, delivery_pos = find_best_insertion_positions_v5(
                    solution, best_vehicle, call, data, vehicle_calls[best_vehicle]
                )
                
                # Update vehicle boundaries (may have changed after previous insertions)
                vehicle_bounds = [i for i, x in enumerate(solution) if x == 0]
                start_idx = 0 if best_vehicle == 0 else vehicle_bounds[best_vehicle - 1] + 1
                end_idx = vehicle_bounds[best_vehicle] if best_vehicle < len(vehicle_bounds) else len(solution)
                
                # If dummy vehicle, just append
                if best_vehicle == n_vehicles - 1:
                    solution.append(call)
                    solution.append(call)
                    vehicle_calls[best_vehicle].add(call)
                    continue
                
                # Empty vehicle case
                if start_idx == end_idx:
                    solution.insert(start_idx, call)
                    solution.insert(start_idx + 1, call)
                    vehicle_calls[best_vehicle].add(call)
                    continue
                
                # Regular case - insert at calculated positions
                # Adjust positions relative to the vehicle's start index
                abs_pickup_pos = start_idx + pickup_pos
                
                # Insert pickup
                solution.insert(abs_pickup_pos, call)
                
                # Update the vehicle boundaries after pickup insertion
                vehicle_bounds = [i for i, x in enumerate(solution) if x == 0]
                start_idx = 0 if best_vehicle == 0 else vehicle_bounds[best_vehicle - 1] + 1
                end_idx = vehicle_bounds[best_vehicle] if best_vehicle < len(vehicle_bounds) else len(solution)
                
                # Adjust delivery position (accounting for the inserted pickup)
                abs_delivery_pos = start_idx + delivery_pos
                if delivery_pos >= pickup_pos:
                    abs_delivery_pos += 1
                
                # Insert delivery
                solution.insert(abs_delivery_pos, call)
                
                # Update the set of calls in this vehicle
                vehicle_calls[best_vehicle].add(call)
    
    return np.array(solution)

def select_best_vehicle_v5(solution, calls_subgroup, data, vehicle_calls):
    """
    Select the best vehicle for a subgroup of calls based on:
    1. Vehicle compatibility with ALL calls in the subgroup
    2. Clustering affinity with existing calls in the vehicle
    3. Vehicle capacity vs call size requirements
    4. Cost optimization (outsource vs transport)
    
    Returns the selected vehicle index.
    """
    n_vehicles = data['n_vehicles']
    
    # Initial eligibility check - must be compatible with ALL calls
    eligible_vehicles = []
    for v in range(n_vehicles):
        if all(data['VesselCargo'][v][call - 1] == 1 for call in calls_subgroup):
            eligible_vehicles.append(v)
    
    if not eligible_vehicles:
        # If no vehicle is compatible with all calls, return dummy vehicle
        return n_vehicles - 1
    
    # Calculate scores for each eligible vehicle
    vehicle_scores = {}
    
    for v in eligible_vehicles:
        if v == n_vehicles - 1:  # Dummy vehicle
            # Base score for dummy based on outsource cost
            avg_outsource_cost = np.mean([data['Cargo'][call - 1, 3] for call in calls_subgroup])
            outsource_factor = 1.0 if avg_outsource_cost < data['average_cost_outsource'] else 0.4
            vehicle_scores[v] = 0.5 * outsource_factor
            continue
            
        # Calculate cluster affinity - how many existing calls share the same cluster
        existing_calls = vehicle_calls[v]
        
        # If vehicle is empty, assign a neutral affinity
        if not existing_calls:
            cluster_affinity = 0.5
        else:
            affinity_count = 0
            for call in calls_subgroup:
                call_cluster = data['cluster_assignments'][call - 1]
                # Count matching cluster calls in this vehicle
                matches = sum(1 for existing in existing_calls 
                             if data['cluster_assignments'][existing - 1] == call_cluster)
                
                affinity_count += min(matches, 3)  # Cap at 3 matches
            
            # Normalize and scale affinity (0.3 to 1.0)
            cluster_affinity = 0.3 + (0.7 * min(affinity_count, len(calls_subgroup) * 2) / 
                                     (len(calls_subgroup) * 2))
        
        # Calculate capacity fit (how well calls fit in vehicle)
        total_call_size = sum(data['Cargo'][call - 1, 2] for call in calls_subgroup)
        capacity_ratio = total_call_size / data['VesselCapacity'][v]
        
        # Score capacity fit - prefer vehicles with good utilization
        if capacity_ratio > 0.9:  # Too tight
            capacity_score = 0.4
        elif capacity_ratio > 0.7:  # Good utilization
            capacity_score = 1.0
        elif capacity_ratio > 0.5:  # Moderate utilization
            capacity_score = 0.8
        else:  # Low utilization
            capacity_score = 0.6
            
        # Calculate route efficiency - approximate cost impact
        route_efficiency = 0.7  # Default neutral score
        
        # Balance score - prefer underused vehicles
        current_load = len(existing_calls) 
        avg_calls_per_vehicle = sum(len(calls) for calls in vehicle_calls) / max(1, sum(1 for calls in vehicle_calls if calls))
        
        if current_load > avg_calls_per_vehicle * 1.2:  # Overloaded
            balance_score = 0.4
        elif current_load < avg_calls_per_vehicle * 0.8:  # Underloaded
            balance_score = 0.9
        else:  # Balanced
            balance_score = 0.7
        
        # Combine all factors - weigh cluster affinity highest
        vehicle_scores[v] = (cluster_affinity * 0.5 + 
                           capacity_score * 0.25 + 
                           route_efficiency * 0.1 + 
                           balance_score * 0.15)
    
    # Convert scores to selection probabilities
    total_score = sum(vehicle_scores.values())
    if total_score == 0:
        # If all scores are 0, use uniform probabilities
        probs = {v: 1.0/len(eligible_vehicles) for v in eligible_vehicles}
    else:
        probs = {v: score/total_score for v, score in vehicle_scores.items()}
    
    # Select vehicle based on probability distribution
    vehicles = list(probs.keys())
    probabilities = [probs[v] for v in vehicles]
    
    return np.random.choice(vehicles, p=probabilities)

def find_best_insertion_positions_v5(solution, vehicle_idx, call, data, existing_calls):
    """
    Find optimal insertion positions for pickup and delivery points based on:
    1. Travel cost minimization
    2. Cluster relationships between calls
    3. Proximity of pickup/delivery nodes to existing nodes
    4. Precedence constraints (pickup before delivery)
    
    Returns (pickup_position, delivery_position) relative to the vehicle route.
    """
    # Get vehicle route bounds
    vehicle_bounds = [i for i, x in enumerate(solution) if x == 0]
    start_idx = 0 if vehicle_idx == 0 else vehicle_bounds[vehicle_idx - 1] + 1
    end_idx = vehicle_bounds[vehicle_idx] if vehicle_idx < len(vehicle_bounds) else len(solution)
    
    # Extract vehicle route
    route = solution[start_idx:end_idx]
    route_len = len(route)
    
    # If dummy vehicle or empty route, return default positions
    if vehicle_idx == data['n_vehicles'] - 1 or route_len == 0:
        return 0, 1
    
    # Get call pickup and delivery nodes
    call_pickup = int(data['Cargo'][call - 1, 0]) - 1
    call_delivery = int(data['Cargo'][call - 1, 1]) - 1
    call_cluster = data['cluster_assignments'][call - 1]
    
    # Initialize position scores with default values
    position_scores_pickup = np.ones(route_len + 1) * 0.1  # Default low score
    position_scores_delivery = np.ones(route_len + 2) * 0.1  # +2 for after pickup insertion
    
    # Map of existing nodes in the route and whether they're pickups or deliveries
    node_positions = {}
    pickup_status = {}  # Track which calls are pickups vs deliveries
    
    for pos, route_call in enumerate(route):
        if route_call == 0:  # Skip separators
            continue
            
        # Determine if this is pickup or delivery
        is_pickup = route_call not in pickup_status
        pickup_status[route_call] = is_pickup
        
        # Get the actual node for this position
        node_idx = int(data['Cargo'][route_call - 1, 0]) - 1 if is_pickup else int(data['Cargo'][route_call - 1, 1]) - 1
        node_positions[pos] = (node_idx, route_call, is_pickup)
    
    # Score pickup positions - consider travel costs
    for pos in range(route_len + 1):
        # Find nodes before and after this position
        prev_node_info = node_positions.get(pos - 1, None) if pos > 0 else None
        next_node_info = node_positions.get(pos, None) if pos < route_len else None
        
        # Default score if we don't have context
        position_scores_pickup[pos] = 0.5
        
        # Score based on travel cost if we have adjacent nodes
        if prev_node_info and next_node_info:
            prev_node, prev_call, prev_is_pickup = prev_node_info
            next_node, next_call, next_is_pickup = next_node_info
            
            # Calculate travel costs - inserting call_pickup between prev_node and next_node
            prev_cost = data['TravelCost'][vehicle_idx, prev_node, call_pickup]
            next_cost = data['TravelCost'][vehicle_idx, call_pickup, next_node]
            direct_cost = data['TravelCost'][vehicle_idx, prev_node, next_node]
            
            # Delta cost for this insertion (lower is better)
            delta_cost = (prev_cost + next_cost - direct_cost) / data['average_travel_cost']
            
            # Convert cost to score (lower cost = higher score)
            cost_score = max(0.1, 1.0 - min(1.0, delta_cost / 2))
            
            # Check cluster relationships
            prev_cluster = data['cluster_assignments'][prev_call - 1]
            next_cluster = data['cluster_assignments'][next_call - 1]
            
            # Boost score for good cluster relationships
            cluster_bonus = 1.0
            if call_cluster == prev_cluster and prev_is_pickup:
                # Same cluster, and appropriate node type
                cluster_bonus += 0.3
            if call_cluster == next_cluster and next_is_pickup:
                # Same cluster, and appropriate node type
                cluster_bonus += 0.3
                
            # Final score combines cost and cluster factors
            position_scores_pickup[pos] = cost_score * cluster_bonus
            
        elif prev_node_info:  # Only prev node exists (end of route)
            prev_node, prev_call, prev_is_pickup = prev_node_info
            
            # Score based on continuation from previous node
            prev_cost = data['TravelCost'][vehicle_idx, prev_node, call_pickup]
            cost_score = max(0.1, 1.0 - min(1.0, prev_cost / data['average_travel_cost']))
            
            # Check cluster relationship with previous call
            if call_cluster == data['cluster_assignments'][prev_call - 1]:
                cost_score *= 1.3
                
            position_scores_pickup[pos] = cost_score
            
        elif next_node_info:  # Only next node exists (start of route)
            next_node, next_call, next_is_pickup = next_node_info
            
            # Score based on how good this is as a first node
            next_cost = data['TravelCost'][vehicle_idx, call_pickup, next_node]
            start_cost = data['FirstTravelCost'][vehicle_idx, call_pickup]
            
            # Combine route start and next node costs
            cost_score = max(0.1, 1.0 - min(1.0, (start_cost + next_cost) / 
                                            (2 * data['average_travel_cost'])))
            
            # Check cluster relationship
            if call_cluster == data['cluster_assignments'][next_call - 1]:
                cost_score *= 1.3
                
            position_scores_pickup[pos] = cost_score
        
    # Now score delivery positions
    for pos in range(route_len + 2):  # +2 because delivery comes after pickup
        # Skip positions before pickup (delivery must come after pickup)
        if pos <= 0:
            position_scores_delivery[pos] = 0.1  # Very low score
            continue
            
        # Convert to position in the original route (accounting for the pickup insertion)
        route_pos = pos - 1
        
        # Get adjacent nodes if they exist
        prev_pos = max(0, route_pos - 1)
        next_pos = min(route_len - 1, route_pos) if route_len > 0 else 0
        
        prev_node_info = node_positions.get(prev_pos, None)
        next_node_info = node_positions.get(next_pos, None)
        
        # Calculate proximity score for delivery
        delivery_score = 0.5  # Default score
        
        if prev_node_info and next_node_info:
            prev_node, prev_call, prev_is_pickup = prev_node_info
            next_node, next_call, next_is_pickup = next_node_info
            
            # Check if prev or next call is in the same cluster
            prev_call_cluster = data['cluster_assignments'][prev_call - 1]
            next_call_cluster = data['cluster_assignments'][next_call - 1]
            
            # Get relationship types
            prev_relationship = data['relationship_type'][call - 1, prev_call - 1]
            next_relationship = data['relationship_type'][call - 1, next_call - 1]
            
            # Calculate travel costs
            prev_proximity = np.mean([data['TravelCost'][v, prev_node, call_delivery] 
                                    for v in range(data['TravelCost'].shape[0])])
            next_proximity = np.mean([data['TravelCost'][v, call_delivery, next_node] 
                                    for v in range(data['TravelCost'].shape[0])])
            
            # Normalize
            prev_proximity = min(1.0, prev_proximity / data['average_travel_cost'])
            next_proximity = min(1.0, next_proximity / data['average_travel_cost'])
            
            # Invert so smaller distances get higher scores
            prev_score = 1.0 - prev_proximity
            next_score = 1.0 - next_proximity
            
            # Boost score if in same cluster
            if prev_call_cluster == call_cluster and prev_relationship == 1 and not prev_is_pickup:
                # Type 1: delivery-delivery relationship
                prev_score *= 1.3
            if next_call_cluster == call_cluster and next_relationship == 1 and not next_is_pickup:
                # Type 1: delivery-delivery relationship
                next_score *= 1.3
                
            # Combine scores
            delivery_score = (prev_score + next_score) / 2
            
        position_scores_delivery[pos] = delivery_score
    
    # Convert scores to probabilities
    pickup_probs = position_scores_pickup / np.sum(position_scores_pickup)
    delivery_probs = position_scores_delivery / np.sum(position_scores_delivery)
    
    # Select positions based on probabilities
    pickup_pos = np.random.choice(range(len(pickup_probs)), p=pickup_probs)
    delivery_pos = np.random.choice(range(len(delivery_probs)), p=delivery_probs)
    
    # Ensure delivery comes after pickup
    if delivery_pos <= pickup_pos:
        delivery_pos = pickup_pos + 1
    
    return pickup_pos, delivery_pos

