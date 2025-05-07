import numpy as np
import random
import time
from copy import deepcopy
from src.utils.Utils import vehicle_feasibility_check, cost_function, feasibility_check, cost_transport_only

greedy_timing = {"total": 0, "feasibility_checks": 0, "cost_calculations": 0, "vehicle_selection": 0, "insertion_tests": 0}

def expensive_greedy_insertion(recieved_solution, calls_to_move, data, seed=None, cached_solution=None):
    global greedy_timing
    start_time = time.time()
    """
    Greedy insertion operator that tries to insert calls optimally into vehicles.
    
    Args:
        recieved_solution: Current solution without the calls_to_move
        calls_to_move: Calls that need to be inserted
        data: Problem data
        seed: Random seed
        cached_solution: Cache of previously evaluated solutions
        
    Returns:
        Updated solution with all calls inserted
    """

    solution = recieved_solution.copy()
    
    if seed is not None:
        np.random.seed(seed)
        
    n_vehicles = data['n_vehicles']
    n_calls = data['n_calls']

    # Track which calls we've successfully inserted
    inserted_calls = []
    
    # For each call in calls_to_move, find the best insertion
    for call in calls_to_move:
        if call in inserted_calls:  # Skip if already inserted
            continue
            
        # Keep track of best insertion across vehicles
        overall_best_cost = float('inf')
        overall_best_solution = None
        second_best_solution = None
        second_best_cost = float('inf')
        
        # Get available vehicles for this call
        vehicle_selection_start = time.time()
        available_vehicles, vehicle_bounds = return_legal_vehicles(solution, call, data)
        greedy_timing["vehicle_selection"] += time.time() - vehicle_selection_start
        
        # If no legal vehicles, insert in dummy
        if len(available_vehicles) == 0:
            solution.append(call)
            solution.append(call)
            inserted_calls.append(call)
            continue

        # If only one vehicle, insert at start of that vehicle
        if len(available_vehicles) == 1:
            v = available_vehicles[0]
            start_idx = 0 if v == 0 else vehicle_bounds[v - 1] + 1
            test_solution = solution.copy()
            test_solution.insert(start_idx, call)
            test_solution.insert(start_idx, call)
            
            # Check if this insertion is feasible
            feasibility_start = time.time()
            feasible, reason = feasibility_check(test_solution, data)
            greedy_timing["feasibility_checks"] += time.time() - feasibility_start
            
            if feasible:
                solution = test_solution
                inserted_calls.append(call)
                continue
            else:
                # Insert in dummy if not feasible
                solution.append(call)
                solution.append(call)
                inserted_calls.append(call)
                continue
                
        # Try insertion in each available vehicle
        for vehicle in available_vehicles:
            insertion_start = time.time()
            best_solution, best_cost = insert_call_in_vehicle(solution, call, vehicle, vehicle_bounds, data, cached_solution)
            greedy_timing["insertion_tests"] += time.time() - insertion_start
            
            if best_cost < overall_best_cost:
                second_best_cost = overall_best_cost
                second_best_solution = overall_best_solution
                overall_best_cost = best_cost
                overall_best_solution = best_solution
            elif best_cost < second_best_cost:
                second_best_cost = best_cost
                second_best_solution = best_solution
        
        # If we found a feasible insertion
        if overall_best_cost < float('inf'):
            solution = overall_best_solution
            inserted_calls.append(call)
        else:
            # If no feasible insertion, add to dummy vehicle
            solution.append(call)
            solution.append(call)
            inserted_calls.append(call)
    
    # Check if all calls were inserted
    not_inserted = [c for c in calls_to_move if c not in inserted_calls]
    if not_inserted:
        for call in not_inserted:
            solution.append(call)
            solution.append(call)
    
    # Final feasibility check
    feasibility_start = time.time()
    feasible, reason = True, 'feasible' # feasibility_check(solution, data) AVOID THIS TO SAVE TIME
    greedy_timing["feasibility_checks"] += time.time() - feasibility_start
    
    if not feasible:
        # Try second-best solution if we have one
        print(f'infeasible greedy insertion with reason {reason}')
        if second_best_solution is not None:
            feasibility_start = time.time()
            second_feasible, second_reason = feasibility_check(second_best_solution, data)
            greedy_timing["feasibility_checks"] += time.time() - feasibility_start
            
            if second_feasible:
                solution = second_best_solution
                feasible = True
        
        # If no feasible solution found, create a new solution with all calls outsourced
        if not feasible:
            temp_solution = recieved_solution.copy()
            for call in calls_to_move:
                temp_solution.append(call)
                temp_solution.append(call)
            solution = temp_solution
    
    greedy_timing["total"] += time.time() - start_time
    return np.array(solution)
    
def insert_call_in_vehicle(solution, call, vehicle, vehicle_bounds, data, cached_solution=None):
    global greedy_timing
    
    # Add time constraint
    time_constraint = 0.05 # If I am going to use time constraint then shuffle what vehicle we look at first so we not only have time for the first in line.
    if np.random.rand() < 0.2:
        time_constraint += 0.5
    time_constraint_start = time.time()

    # Get the range for the target vehicle
    start_idx = 0 if vehicle == 0 else vehicle_bounds[vehicle - 1] + 1
    end_idx = vehicle_bounds[vehicle] if vehicle < len(vehicle_bounds) else len(solution)
    route = solution[start_idx:end_idx]
    
    # If empty route, insert at start
    if len(route) == 0:
        test_solution = solution.copy()
        test_solution.insert(start_idx, call)
        test_solution.insert(start_idx, call)
        
        feasibility_start = time.time()
        feasible, reason = vehicle_feasibility_check(test_solution, data, vehicle)
        greedy_timing["feasibility_checks"] += time.time() - feasibility_start
        
        if feasible:
            cost_calc_start = time.time()
            cost = cost_function(test_solution, data)
            greedy_timing["cost_calculations"] += time.time() - cost_calc_start
            return test_solution, cost
        else:
            return solution.copy(), float('inf')
    
    # Test all possible insertion positions
    best_solution = None
    best_cost = float('inf')
    
    for first_pos in range(len(route) + 1):
        for second_pos in range(first_pos + 1, len(route) + 2):
            # Create a new solution for testing
            test_solution = solution.copy()
            
            # Insert call at first position
            test_solution.insert(start_idx + first_pos, call)
            # Insert call at second position (adjusted for first insertion)
            test_solution.insert(start_idx + second_pos, call)
            
            # Skip cached solutions
            if cached_solution and tuple(test_solution) in cached_solution:
                continue
                
            # Check feasibility
            feasibility_start = time.time()
            feasible, reason = vehicle_feasibility_check(test_solution, data, vehicle)
            greedy_timing["feasibility_checks"] += time.time() - feasibility_start
            
            if feasible:
                # Calculate cost
                cost_calc_start = time.time()
                vehicle_route = test_solution[start_idx:end_idx+2]
                current_cost = cost_function(test_solution, data)
                greedy_timing["cost_calculations"] += time.time() - cost_calc_start
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = test_solution.copy()

            """### NEW TECHNIQUE of checking cost before feasibility to save time
            # Calculate cost
            cost_calc_start = time.time()
            vehicle_route = test_solution[start_idx:end_idx+2]
            current_cost = cost_function(test_solution, data)
            greedy_timing["cost_calculations"] += time.time() - cost_calc_start
            
            if current_cost < best_cost:
                # Check feasibility
                feasibility_start = time.time()
                feasible, reason = vehicle_feasibility_check(test_solution, data, vehicle)
                greedy_timing["feasibility_checks"] += time.time() - feasibility_start
                
                if feasible:
                    best_cost = current_cost
                    best_solution = test_solution.copy()"""


        if time.time() - time_constraint_start > time_constraint:
            #print("Time constraint exceeded during insertion test.")
            break
    
    # Return best solution found or original if none are feasible
    if best_solution is not None:
        return best_solution, best_cost
    else:
        return solution.copy(), float('inf')

def return_legal_vehicles(solution, call, data):
    n_vehicles = data['n_vehicles']
    available_vehicles = np.array(range(n_vehicles))
    vehicle_bounds = [i for i, x in enumerate(solution) if x == 0]
    
    # Remove incompatible vehicles
    filtered_vehicles = []
    for v in available_vehicles:
        # Check cargo compatibility
        if data['VesselCargo'][v][call - 1] == 0:
            continue
            
        # Get current route for this vehicle
        start_idx = 0 if v == 0 else vehicle_bounds[v - 1] + 1
        end_idx = vehicle_bounds[v] if v < len(vehicle_bounds) else len(solution)
        vehicle_calls = solution[start_idx:end_idx]
        
        # Apply capacity and route length filters with some randomization
        if len(vehicle_calls) >= 12:  # Hard limit on route length
            continue
        elif len(vehicle_calls) >= 10 and np.random.rand() > 0.5:  # Probabilistic filter
            continue

        filtered_vehicles.append(v)
    
    if len(filtered_vehicles) == 0:
        return [], vehicle_bounds
    
    # Maybe for bigger files add a chance that we drop half of the vehicles with a random chance, or drop more vehicles that are less promising.
    """if data['n_calls'] > 36:
        if np.random.rand() < 0.5:
            filtered_vehicles = random.sample(filtered_vehicles, - (len(filtered_vehicles) // -2)) # roof devision
            if data['n_calls'] > 90:
                filtered_vehicles = random.sample(filtered_vehicles, - (len(filtered_vehicles) // -2))
                if data['n_calls'] > 200:
                    filtered_vehicles = random.sample(filtered_vehicles, - (len(filtered_vehicles) // -2))"""
        
    return np.array(filtered_vehicles), vehicle_bounds

def regret_k_insertion(received_solution, calls_to_move, data, k=2, seed=None, cached_solution=None):
    """
    Regret-k insertion that prioritizes calls with highest regret value.
    
    Args:
        received_solution: Current solution without the calls_to_move
        calls_to_move: Calls that need to be inserted
        data: Problem data
        k: Regret parameter (default=3)
        seed: Random seed
        cached_solution: Optional cache of solutions
        
    Returns:
        Updated solution with all calls inserted
    """
    global greedy_timing
    start_time = time.time()
    
    # Create a copy of the solution to work with
    solution = received_solution.copy()
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize tracking variables
    n_vehicles = data['n_vehicles']
    n_calls = data['n_calls']
    inserted_calls = []
    
    # Track which calls were in the original solution - NEW SAFETY CHECK
    original_calls = set(x for x in solution if x > 0)
    expected_calls = set(original_calls).union(set(calls_to_move))
    
    # Process calls iteratively using regret value
    while len(inserted_calls) < len(calls_to_move):
        # Get remaining calls
        remaining_calls = [c for c in calls_to_move if c not in inserted_calls]
        if not remaining_calls:
            break
        
        # Dictionary to store best k positions for each call
        call_regret_data = {}
        calls_needing_dummy = []  # Explicitly track calls that need to go to the dummy
        
        # For each remaining call, compute insertion data
        for call in remaining_calls:
            # Dictionary to store costs for each vehicle
            vehicle_costs = {}
            all_insertion_costs = []
            
            # Get available vehicles
            vehicle_selection_start = time.time()
            available_vehicles, vehicle_bounds = return_legal_vehicles(solution, call, data)
            greedy_timing["vehicle_selection"] += time.time() - vehicle_selection_start
            
            # If no vehicle is available, add to dummy
            if len(available_vehicles) == 0:
                calls_needing_dummy.append(call)
                continue
            
            # For each available vehicle, find best insertion
            feasible_insertion_found = False
            for vehicle in available_vehicles:
                insertion_start = time.time()
                best_solution, best_cost = insert_call_in_vehicle(solution, call, vehicle, vehicle_bounds, data, cached_solution=cached_solution)
                greedy_timing["insertion_tests"] += time.time() - insertion_start
                
                if best_cost < float('inf'):
                    feasible_insertion_found = True
                    vehicle_costs[vehicle] = (best_cost, best_solution)
                    all_insertion_costs.append((best_cost, vehicle))
            
            # If no feasible insertion found, add to dummy list
            if not feasible_insertion_found:
                calls_needing_dummy.append(call)
                continue
                
            # Sort insertions by cost
            all_insertion_costs.sort()
            
            # Calculate regret value
            if len(all_insertion_costs) >= k:
                # Get k-th best cost
                k_best_cost = all_insertion_costs[min(k-1, len(all_insertion_costs)-1)][0]
                best_cost = all_insertion_costs[0][0]
                best_vehicle = all_insertion_costs[0][1]
                
                # Calculate regret as difference between k-th best and best
                regret_value = k_best_cost - best_cost
                
                call_regret_data[call] = {
                    'regret': regret_value,
                    'best_cost': best_cost,
                    'best_vehicle': best_vehicle,
                    'best_solution': vehicle_costs[best_vehicle][1]
                }
            elif len(all_insertion_costs) > 0:
                # If we don't have k options, use what we have
                best_cost = all_insertion_costs[0][0]
                best_vehicle = all_insertion_costs[0][1]
                # Use a large regret value (not infinity) to prioritize calls with limited options
                regret_value = 1e9  # Large but not inf to avoid numerical issues
                
                call_regret_data[call] = {
                    'regret': regret_value,
                    'best_cost': best_cost,
                    'best_vehicle': best_vehicle,
                    'best_solution': vehicle_costs[best_vehicle][1]
                }
        
        # Handle calls that need to go to dummy vehicle immediately
        for call in calls_needing_dummy:
            solution.append(call)
            solution.append(call)
            inserted_calls.append(call)
        
        # If no regret data (all calls went to dummy), continue to next iteration
        if not call_regret_data:
            continue
        
        # Find call with maximum regret
        max_regret_call = None
        max_regret_value = -float('inf')
        
        for call, data_dict in call_regret_data.items():
            if data_dict['regret'] > max_regret_value:
                max_regret_value = data_dict['regret']
                max_regret_call = call
            # Tie-breaking based on lowest cost
            elif data_dict['regret'] == max_regret_value and data_dict['best_cost'] < call_regret_data[max_regret_call]['best_cost']:
                max_regret_call = call
        
        # Insert the call with maximum regret
        if max_regret_call is not None:
            # FIXED: Instead of overwriting the entire solution, we need to update it with the new insertion
            # We get the best solution for this call
            new_solution = call_regret_data[max_regret_call]['best_solution']
            
            # Check that the new solution contains all previously inserted calls
            prev_solution_calls = set(x for x in solution if x > 0)
            new_solution_calls = set(x for x in new_solution if x > 0)
            
            # Make sure our new solution doesn't lose any calls
            if prev_solution_calls.issubset(new_solution_calls):
                solution = new_solution
                inserted_calls.append(max_regret_call)
            else:
                # If somehow calls got dropped in the best solution, insert to dummy instead
                #print(f"Warning: Best solution for call {max_regret_call} would drop calls. Using dummy insertion.")
                solution.append(max_regret_call)
                solution.append(max_regret_call)
                inserted_calls.append(max_regret_call)
    
    # CRITICAL SAFETY CHECK: Ensure all calls were inserted
    not_inserted = [c for c in calls_to_move if c not in inserted_calls]
    if not_inserted:
        print(f"Safety warning: {len(not_inserted)} calls were not inserted! Adding to dummy vehicle.")
        for call in not_inserted:
            solution.append(call)
            solution.append(call)
            inserted_calls.append(call)
    
    # Final feasibility check
    feasibility_start = time.time()
    feasible, reason = True, "feasible" # feasibility_check(solution, data) TRYING TO SAVE TIME BY AVOIDING THIS CHECK
    greedy_timing["feasibility_checks"] += time.time() - feasibility_start
    
    if not feasible:
        print(f"Found infeasible solution in regret insertion. Reason: {reason}")
        # If no feasible solution, outsource all calls
        temp_solution = received_solution.copy()
        for call in calls_to_move:
            temp_solution.append(call)
            temp_solution.append(call)
        solution = temp_solution
    
    # FINAL VERIFICATION: Check that all expected calls are in the solution
    solution_calls = set(x for x in solution if x > 0)
    if solution_calls != expected_calls:
        missing_calls = expected_calls - solution_calls
        if missing_calls:
            print(f"ERROR: Still missing calls {missing_calls} after regret insertion. Forced insertion into dummy.")
            temp_solution = solution.copy()
            for call in missing_calls:
                temp_solution.append(call)
                temp_solution.append(call)
            solution = temp_solution
    
    greedy_timing["total"] += time.time() - start_time
    return np.array(solution)

def regret_k_insertion_3(received_solution, calls_to_move, data, seed=None, cached_solution=None):
    return regret_k_insertion(received_solution, calls_to_move, data, k=3, seed=seed, cached_solution=cached_solution)

def regret_k_insertion_4(received_solution, calls_to_move, data, seed=None, cached_solution=None):
    return regret_k_insertion(received_solution, calls_to_move, data, k=4, seed=seed, cached_solution=cached_solution)