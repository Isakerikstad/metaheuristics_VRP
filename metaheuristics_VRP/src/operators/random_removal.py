import numpy as np
from collections import defaultdict

# Global tracking dictionary to store call removal statistics
call_removal_tracker = {
    'individual_calls': defaultdict(int),  # Counts how often each call is removed
    'call_pairs': defaultdict(int),        # Counts how often each call pair appears together
    'call_triplets': defaultdict(int),     # Counts how often each call triplet appears together
    'iteration_count': 0                   # Total number of times removal operators have been called
}

def reset_call_tracker():
    """Reset the call tracker statistics."""
    global call_removal_tracker
    call_removal_tracker = {
        'individual_calls': defaultdict(int),
        'call_pairs': defaultdict(int),
        'call_triplets': defaultdict(int),
        'iteration_count': 0
    }

def update_call_tracker(calls_to_move):
    """Update the call tracker with the current list of removed calls."""
    global call_removal_tracker
    
    # Increment iteration count
    call_removal_tracker['iteration_count'] += 1
    
    # Update individual call counts
    for call in calls_to_move:
        call_removal_tracker['individual_calls'][call] += 1
    
    # Update pair counts
    if len(calls_to_move) >= 2:
        for i in range(len(calls_to_move)):
            for j in range(i + 1, len(calls_to_move)):
                # Ensure consistent ordering for the pair key
                call_pair = tuple(sorted([calls_to_move[i], calls_to_move[j]]))
                call_removal_tracker['call_pairs'][call_pair] += 1
    
    # Update triplet counts
    if len(calls_to_move) >= 3:
        for i in range(len(calls_to_move)):
            for j in range(i + 1, len(calls_to_move)):
                for k in range(j + 1, len(calls_to_move)):
                    # Ensure consistent ordering for the triplet key
                    call_triplet = tuple(sorted([calls_to_move[i], calls_to_move[j], calls_to_move[k]]))
                    call_removal_tracker['call_triplets'][call_triplet] += 1


def random_removal(solution, data, seed=None, min_remove=1, max_remove=15):
    if seed is not None:
        np.random.seed(seed)
    
    # Convert to list if numpy array
    if isinstance(solution, np.ndarray):
        solution = solution.tolist()
    
    # Find all calls in the solution (excluding 0s)
    calls = set(x for x in solution if x > 0)
    if not calls:
        return solution, []
    
    calls_to_move = []

    # Randomly select between 1 and 4 calls to remove and reinsert
    for a in range(np.random.randint(min_remove, max_remove + 1)):
        # Ensure we don't select more calls than available
        if len(calls) == 0:
            break

        calls_to_move.append(np.random.choice(list(calls)))

        # Remove the selected call from the set
        calls.remove(calls_to_move[a])
    
        # Find positions of the selected call
        pickup_idx = solution.index(calls_to_move[a])
        delivery_idx = len(solution) - 1 - solution[::-1].index(calls_to_move[a])
        
        # Remove the call from both positions
        solution.pop(delivery_idx)
        solution.pop(pickup_idx)
    
    # Update the call tracker with this removal
    update_call_tracker(calls_to_move)

    return solution, calls_to_move
    

def random_removal_clipped(solution, data, seed=None, min_remove=3, max_remove=30):
    if seed is not None:
        np.random.seed(seed)
    
    # Convert to list if numpy array
    if isinstance(solution, np.ndarray):
        solution = solution.tolist()
    
    # Find all calls in the solution (excluding 0s)
    calls = set(x for x in solution if x > 0)
    if not calls:
        return solution, []
    
    calls_to_move = []

    # Randomly select between min_remove and max_remove calls to remove and reinsert
    for a in range(np.random.randint(min_remove, max_remove + 1)):
        # Ensure we don't select more calls than available
        if len(calls) == 0:
            break

        calls_to_move.append(np.random.choice(list(calls)))

        # Remove the selected call from the set
        calls.remove(calls_to_move[a])
    
        # Find positions of the selected call
        pickup_idx = solution.index(calls_to_move[a])
        delivery_idx = len(solution) - 1 - solution[::-1].index(calls_to_move[a])
        
        # Remove the call from both positions
        solution.pop(delivery_idx)
        solution.pop(pickup_idx)
    
    # Update the call tracker with this removal
    update_call_tracker(calls_to_move)

    return solution, calls_to_move


def exploration_removal(solution, data, seed=None, min_remove=1, max_remove=25):
    """
    Removal operator that prioritizes less-explored calls and combinations.
    Uses the call_removal_tracker to favor calls that have been removed less frequently.
    """
    global call_removal_tracker
    
    if seed is not None:
        np.random.seed(seed)
    
    # Convert to list if numpy array
    if isinstance(solution, np.ndarray):
        solution = solution.tolist()
    
    # Find all calls in the solution (excluding 0s)
    calls = list(set(x for x in solution if x > 0))
    if not calls:
        return solution, []
    
    # Calculate weights - inverse of removal frequency + small constant to avoid zeros
    # Higher weight means less frequently removed (more desirable for exploration)
    call_weights = []
    total_iterations = max(1, call_removal_tracker['iteration_count'])  # Avoid division by zero
    
    for call in calls:
        call_count = call_removal_tracker['individual_calls'].get(call, 0)
        # Normalized removal frequency (0 to 1 range)
        frequency = call_count / total_iterations
        # Inverse to prioritize less removed calls, with a small constant to avoid division by zero
        weight = 1.0 / (frequency + 0.01)
        call_weights.append(weight)
    
    # Normalize weights
    sum_weights = sum(call_weights)
    if sum_weights > 0:
        call_weights = [w / sum_weights for w in call_weights]
    else:
        call_weights = [1.0 / len(calls) for _ in calls]
    
    # Determine number of calls to remove
    num_to_remove = np.random.randint(min_remove, min(max_remove + 1, len(calls) + 1))
    calls_to_move = []
    
    # Select calls using weighted random choice
    remaining_calls = calls.copy()
    remaining_weights = call_weights.copy()
    
    for _ in range(num_to_remove):
        if not remaining_calls:
            break
            
        # Choose a call using weighted probabilities
        selected_idx = np.random.choice(len(remaining_calls), p=remaining_weights)
        selected_call = remaining_calls[selected_idx]
        calls_to_move.append(selected_call)
        
        # Remove from lists for next iteration
        remaining_calls.pop(selected_idx)
        remaining_weights.pop(selected_idx)
        
        # Re-normalize remaining weights
        if remaining_weights:
            sum_weights = sum(remaining_weights)
            if sum_weights > 0:
                remaining_weights = [w / sum_weights for w in remaining_weights]
    
    # Remove the selected calls from the solution
    solution_copy = solution.copy()
    solution = []
    
    # Build a new solution excluding the calls we want to remove
    for call in solution_copy:
        if call in calls_to_move and solution_copy.count(call) > solution.count(call):
            continue  # Skip this occurrence of the call
        solution.append(call)
    
    # Update the call tracker with this removal
    update_call_tracker(calls_to_move)
    
    return solution, calls_to_move