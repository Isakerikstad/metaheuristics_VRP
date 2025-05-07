import numpy as np
import time

from src.utils.Utils import feasibility_check, cost_function

# Import removal operators
from src.operators.random_removal import random_removal as ran3
from src.operators.random_removal import random_removal_clipped as ran3_clipped
from src.operators.random_removal import exploration_removal as explore_rem
from src.operators.random_removal import reset_call_tracker
from src.operators.shaw_removal import shaw_removal as srm2
from src.operators.shaw_removal import shaw_removal_clipped as srm2_clipped
from src.operators.remove_highest_cost_call import remove_highest_cost_call as rhc4
from src.operators.remove_highest_cost_call import remove_highest_cost_call_clipped as rhc4_clipped

# Import insertion operators
from src.operators.greedy_insertion import expensive_greedy_insertion as gre1
from src.operators.greedy_insertion import regret_k_insertion as reg2
from src.operators.greedy_insertion import regret_k_insertion_3 as regk3
from src.operators.greedy_insertion import regret_k_insertion_4 as regk4

from src.operators.vehicle_swap import vehicle_swap

def final_alg(solution, data, seed=None, cached_solution={}):
    """
    Enhanced Adaptive Weights algorithm incorporating exploration phases
    
    Args:
        solution: Initial solution
        data: Problem data
        seed: Random seed
        cached_solution: Dictionary to cache solutions
        
    Returns:
        best_solution: Best solution found
        feasible_tally: Number of feasible solutions evaluated
        cached_solution: Updated cache of solutions
        best_iteration: Iteration at which best solution was found
    """
    # Reset call tracker
    reset_call_tracker()

    # Define operators
    
    insert_operators = [gre1]
    remove_operators = [ran3, srm2_clipped, rhc4, ran3_clipped, explore_rem]

    time_use = {}
    for op in remove_operators + insert_operators:
        time_use[op.__name__] = [0.0, 0] # [total_time, count]

    if seed is not None:
        np.random.seed(seed)

    incumbent = solution.copy()
    best_solution = solution.copy()
    cost_inc = cost_function(incumbent, data)
    best_solution_cost = cost_inc
    cached_solution[tuple(incumbent)] = [cost_inc, True, 1]
    delta_w = []
    all_iterations = 10000
    segment_size = 100
    
    # Initialize weights and scores for operators
    num_remove_operators = len(remove_operators)
    num_insert_operators = len(insert_operators)
    remove_weights = np.ones(num_remove_operators) / num_remove_operators
    insert_weights = np.ones(num_insert_operators) / num_insert_operators
    remove_scores = np.zeros(num_remove_operators)
    insert_scores = np.zeros(num_insert_operators)
    remove_usage = np.zeros(num_remove_operators, dtype=int)
    insert_usage = np.zeros(num_insert_operators, dtype=int)
    min_weight = 0.05
    max_weight = 0.8
    
    # Performance tracking
    feasible_tally = 1
    iterations_since_improvement = 0
    if data['n_calls'] < 50:
        escape_threshold = 400
    else:
        escape_threshold = 150
    
    # For tracking best solution iteration
    cost_history = np.full(all_iterations, np.inf)
    cost_history[0] = cost_inc
    best_solution_history = np.zeros(all_iterations)
    best_solution_history[0] = best_solution_cost
    accepted_count = 0
    
    # Exploration phase settings
    exploration_phase_iterations = 100  # How many iterations to run in exploration mode
    exploration_phase_frequency = 500   # After how many iterations to trigger exploration phase
    exploration_next_phase = exploration_phase_frequency  # When to start the first exploration phase
    in_exploration_phase = False  # Whether we're currently in an exploration phase
    exploration_counter = 0  # Counter for tracking exploration phase iterations
    
    # Initial warm-up phase (100 iterations)
    time_main_start = time.time()
    for i in range(100):
        iteration = i  # Offset for initialization iterations
        iteration_seed = 100_000 * seed + iteration if seed is not None else None

        # Register time for each remove operator
        time_rem_start = time.time()
        
        # Select remove operator with uniform probability during warm-up
        remove_op_idx = 0 # np.random.randint(0, num_remove_operators - 2)  # Exclude exploration and ran3_clipped
        remove_usage[remove_op_idx] += 1
        
        # Apply removal operator
        temp_solution, removed_calls = remove_operators[remove_op_idx](incumbent, data, seed=iteration_seed)

        time_use[remove_operators[remove_op_idx].__name__][0] += time.time() - time_rem_start
        time_use[remove_operators[remove_op_idx].__name__][1] += 1
        
        # Register time for each insert operator
        time_ins_start = time.time()

        # Select insertion operator with uniform probability during warm-up
        insert_op_idx = np.random.randint(0, num_insert_operators)
        insert_usage[insert_op_idx] += 1
        
        # Apply insertion operator
        new_solution = insert_operators[insert_op_idx](temp_solution, removed_calls, data, cached_solution=cached_solution, seed=iteration_seed)
        
        time_use[insert_operators[insert_op_idx].__name__][0] += time.time() - time_ins_start
        time_use[insert_operators[insert_op_idx].__name__][1] += 1

        # Standardize solution representation
        new_solution = sort_dummy(new_solution)

        # Check feasibility and cost
        if tuple(new_solution) in cached_solution:
            new_cost, feasible, new_counter = cached_solution[tuple(new_solution)]
            new_counter += 1
            cached_solution[tuple(new_solution)] = [new_cost, feasible, new_counter]
        else: 
            feasible, _ = feasibility_check(new_solution, data)
            new_cost = cost_function(new_solution, data)
            cached_solution[tuple(new_solution)] = [new_cost, feasible, 1]
        
        if feasible:
            feasible_tally += 1
            delta_e = new_cost - cost_inc
            
            # Score the operators based on solution quality
            if new_cost < best_solution_cost:
                # New global best (4 points)
                remove_scores[remove_op_idx] += 4
                insert_scores[insert_op_idx] += 4
                best_solution = new_solution.copy()
                best_solution_cost = new_cost
                iterations_since_improvement = 0
                #print(f"New best solution found at iteration {iteration}: {cost_inc}")
            elif delta_e < 0:
                # Better than current solution (2 points)
                remove_scores[remove_op_idx] += 2
                insert_scores[insert_op_idx] += 2
            else:
                # New feasible solution (1 point)
                if cached_solution[tuple(new_solution)][2] == 1:  # First time seeing this solution
                    remove_scores[remove_op_idx] += 1
                    insert_scores[insert_op_idx] += 1
            
            # Simulated annealing acceptance during warm-up
            if delta_e < 0 or np.random.random() < 0.8:  # Accept with high probability during warm-up
                accepted_count += 1
                incumbent = new_solution.copy()
                cost_inc = new_cost
            
            delta_w.append(abs(delta_e))
        
        # Update cost history
        cost_history[iteration] = cost_inc
        best_solution_history[iteration] = best_solution_cost

    # Calculate initial temperature and cooling rate
    delta_avg = np.mean(delta_w) if delta_w else 1.0
    T0 = max(-delta_avg / np.log(0.8), 1.0)
    Tf = 0.01  # Final temperature
    alpha = np.power(Tf / T0, 1.0 / all_iterations)
    T = T0
    iterations_since_improvement = 0
    escape = 0
    incumbent = best_solution.copy()
    
    print(f"Time the operators took to run: {time_use}")

    # Main loop (9900 iterations)
    for i in range(all_iterations - 100):
        
        if time.time() - time_main_start > 1500:
            print(f"25 minute time limit exceeded at iteration {i + 100} \n Best solution: {best_solution_cost} {best_solution}")
            break

        iteration = i + 100  # Offset for initialization + warm-up iterations
        iteration_seed = 100_000 * seed + iteration if seed is not None else None
        
        # Check if we should start an exploration phase
        if not in_exploration_phase and iteration >= exploration_next_phase:
            in_exploration_phase = True
            exploration_counter = 0
        
        # Check if we should end an exploration phase
        if in_exploration_phase and exploration_counter >= exploration_phase_iterations:
            in_exploration_phase = False
            exploration_next_phase = iteration + exploration_phase_frequency
        
        # Check if escape condition is met
        if iterations_since_improvement >= escape_threshold:
            escape += 1
            print(f"Escape condition met at iteration {iteration}, counter: {escape}")
            iterations_since_improvement = 0
            # In escape situation, give more weight to ran3_clipped
            if escape >= 1:
                remove_weights = np.ones(num_remove_operators) * min_weight
                remove_weights[-2] = max_weight  # Boost ran3_clipped
                remove_weights = remove_weights / np.sum(remove_weights)
                if escape == 2:
                    incumbent = best_solution.copy()
                    cost_inc = best_solution_cost
                    print(f"Resetting incumbent to best solution at iteration {iteration}")
                if escape == 3:
                    print(f"Using remove 50 calls operator for escape at iteration {iteration}")
                    for i in range(10):
                        temp_solution, calls = remove_operators[0](incumbent, data, seed=iteration_seed, min_remove=45 + (i * 3), max_remove=55 + (i * 3))
                        # Apply the greedy insertion operator
                        new_solution = insert_operators[0](temp_solution, calls, data, seed=iteration_seed)
                        # Standardize solution representation
                        new_solution = sort_dummy(new_solution)
                        # Check feasibility and cost
                        if feasibility_check(new_solution, data):
                            incumbent = new_solution.copy()
                            cost_inc = cost_function(incumbent, data)
                            if cost_inc < best_solution_cost:
                                print(f"Escape solution found at iteration {iteration}: {cost_inc} {incumbent}")
                                break
                if escape == 4:
                    incumbent = best_solution.copy()
                    cost_inc = best_solution_cost
                    print(f"Resetting incumbent to best solution at iteration {iteration}. Also introducing regret k insertion operator")
                    insert_operators = [gre1, reg2, regk3]
                    insert_weights = np.ones(len(insert_operators)) / len(insert_operators)
                    insert_scores = np.zeros(len(insert_operators))
                    num_insert_operators = len(insert_operators)
                    insert_usage = np.zeros(num_insert_operators, dtype=int)
        
        # Start of a new segment: update weights based on scores
        if i % segment_size == 0 and i > 0:
            if i % (segment_size * 3) == 0:
                print(f"Working on iteration {iteration}. current best: {best_solution_cost} {best_solution}")

            # Try to use vehicle swap
            swap_number = np.random.randint(2, 4)
            temp_incumebent = vehicle_swap(incumbent, data, seed=iteration_seed, num_vehicles_to_swap=swap_number)
            cost_temp_incumebent = cost_function(temp_incumebent, data)
            if cost_temp_incumebent < cost_inc:
                incumbent = temp_incumebent.copy()
                cost_inc = cost_temp_incumebent     
                if cost_temp_incumebent < best_solution_cost:
                    print(f"Vehicle swap improved best solution -> {cost_temp_incumebent} {temp_incumebent}")
                    best_solution = temp_incumebent.copy()
                    best_solution_cost = cost_temp_incumebent

            # Update weights based on scores with smoothing factor, but keep them above min_weight and below max_weight
            if np.sum(remove_scores) > 0:
                remove_weights = np.clip(remove_weights + 0.1 * (remove_scores / np.sum(remove_scores) - remove_weights), min_weight, max_weight)
            if np.sum(insert_scores) > 0:
                insert_weights = np.clip(insert_weights + 0.1 * (insert_scores / np.sum(insert_scores) - insert_weights), min_weight, max_weight)
            
            # Normalize weights
            remove_weights = remove_weights / np.sum(remove_weights)
            insert_weights = insert_weights / np.sum(insert_weights)
            
            # Reset scores for next segment
            remove_scores = np.zeros(num_remove_operators)
            insert_scores = np.zeros(num_insert_operators)

        # Choose operators based on phase
        if in_exploration_phase:
            # Exploration phase - use exploration removal operator
            temp_solution, removed_calls = explore_rem(incumbent, data, seed=iteration_seed)
            remove_usage[-1] += 1  # Using exploration removal (index 6)
            exploration_counter += 1
        else:
            # Normal phase - select remove operator based on adaptive weights
            # If we're in escape mode, we've already adjusted weights to favor certain operators
            remove_op_idx = np.random.choice(range(num_remove_operators), p=remove_weights)
            remove_usage[remove_op_idx] += 1
            temp_solution, removed_calls = remove_operators[remove_op_idx](incumbent, data, seed=iteration_seed)
        
        # Select insertion operator based on adaptive weights
        insert_op_idx = np.random.choice(range(num_insert_operators), p=insert_weights)
        insert_usage[insert_op_idx] += 1
        
        new_solution = insert_operators[insert_op_idx](temp_solution, removed_calls, data, seed=iteration_seed)
        
        # Standardize solution representation
        new_solution = sort_dummy(new_solution)

        # Check feasibility and cost
        if tuple(new_solution) in cached_solution:
            new_cost, feasible, new_counter = cached_solution[tuple(new_solution)]
            new_counter += 1
            cached_solution[tuple(new_solution)] = [new_cost, feasible, new_counter]
        else:                
            feasible = True # , _ = feasibility_check(new_solution, data) CHANGE LATER MAYBE
            new_cost = cost_function(new_solution, data)
            cached_solution[tuple(new_solution)] = [new_cost, feasible, 1]
        
        if feasible:
            feasible_tally += 1
            delta_e = new_cost - cost_inc
            
            # Update best solution if better
            if new_cost < best_solution_cost:
                # New global best (4 points)
                if not in_exploration_phase:
                    remove_scores[remove_op_idx] += 4
                    insert_scores[insert_op_idx] += 4
                best_solution = new_solution.copy()
                best_solution_cost = new_cost
                iterations_since_improvement = 0

                temp_best_solution = vehicle_swap(best_solution, data, seed=iteration_seed) # Try to vehicle swap new best
                cost_temp_best_solution = cost_function(temp_best_solution, data)
                if cost_temp_best_solution < best_solution_cost:
                    print(f"Vehicle swap improved new best solution found at iteration {iteration}: {cost_temp_best_solution} {temp_best_solution}")
                    best_solution = temp_best_solution.copy()
                    best_solution_cost = cost_temp_best_solution
                else:
                    print(f"New best solution found at iteration {iteration}: {best_solution_cost} {best_solution}")

            elif delta_e < 0:
                # Better than current solution (2 points)
                if not in_exploration_phase:
                    remove_scores[remove_op_idx] += 2
                    insert_scores[insert_op_idx] += 2
                iterations_since_improvement = 0
            else:
                # New feasible solution (1 point)
                if cached_solution[tuple(new_solution)][2] == 1:  # First time seeing this solution
                    if not in_exploration_phase:
                        remove_scores[remove_op_idx] += 1
                        insert_scores[insert_op_idx] += 1
                iterations_since_improvement += 1
            
            # Simulated annealing acceptance
            if delta_e > 0:  # Worsening move
                acceptance_prob = np.exp(-delta_e / max(T, 1e-10))
                
                if np.random.random() < acceptance_prob:
                    accepted_count += 1
                    incumbent = new_solution.copy()
                    cost_inc = new_cost
            else:  # Improving move
                accepted_count += 1
                incumbent = new_solution.copy()
                cost_inc = new_cost
                
        else:
            iterations_since_improvement += 1
        
        # Update temperature
        T = max(alpha * T, Tf)
        
        # Update cost history 
        cost_history[iteration] = cost_inc
        best_solution_history[iteration] = best_solution_cost

    # Print timing statistics
    from src.operators.greedy_insertion import greedy_timing
    print(f"Greedy Insertion Timing:")
    print(f"  Total time: {greedy_timing['total']:.2f} seconds")
    print(f"  Feasibility checks: {greedy_timing['feasibility_checks']:.2f} seconds")
    print(f"  Cost calculations: {greedy_timing['cost_calculations']:.2f} seconds")
    print(f"  Vehicle selection: {greedy_timing['vehicle_selection']:.2f} seconds")
    print(f"  Insertion tests: {greedy_timing['insertion_tests']:.2f} seconds")
    
    # Print removal statistics if available
    try:
        from src.operators.random_removal import call_removal_tracker
        print("\nRemoval statistics after run:")
        print(f"Total iterations: {call_removal_tracker['iteration_count']}")
        
        # Find least and most frequently removed calls
        calls = list(call_removal_tracker['individual_calls'].keys())
        call_counts = [call_removal_tracker['individual_calls'][c] for c in calls]
        
        if calls:
            min_count = min(call_counts)
            max_count = max(call_counts)
            least_removed = [c for c, count in call_removal_tracker['individual_calls'].items() if count == min_count]
            most_removed = [c for c, count in call_removal_tracker['individual_calls'].items() if count == max_count]
            
            print(f"Least frequently removed calls: {least_removed} (removed {min_count} times)")
            print(f"Most frequently removed calls: {most_removed} (removed {max_count} times)")
    except Exception as e:
        print(f"Error accessing call removal statistics: {e}")
    
    return best_solution, feasible_tally, cached_solution, np.argmin(best_solution_history)

def sort_dummy(solution):
    """Sort the calls of the dummy vehicle so similar solutions are not unique"""

    zero_indices = np.where(solution == 0)[0]
    dummy_vehicle_ind = zero_indices[-1]
    dummy_calls = solution[dummy_vehicle_ind+1:]
    if len(dummy_calls) > 0:
        dummy_calls.sort()
        solution[dummy_vehicle_ind+1:] = dummy_calls
    return solution

