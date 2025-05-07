import argparse
import time
import numpy as np
import multiprocessing
from src.utils.Utils import load_problem, feasibility_check, cost_function, all_calls_outsorced
from src.algorithms.final_alg import final_alg

file_paths =['data\Call_7_Vehicle_3.txt', 'data\Call_18_Vehicle_5.txt', 'data\Call_35_Vehicle_7.txt', 'data\Call_80_Vehicle_20.txt', 'data\Call_130_Vehicle_40.txt']

# Process a single problem instance
def process_instance(file_path, args):
    local_args = argparse.Namespace(**vars(args))
    local_args.file_path = file_path
    
    # Load problem data
    print(f"Loading problem from {file_path}...")
    data = load_problem(file_path)
    
    # Print problem statistics
    print(f"Problem size: {data['n_calls']} calls, {data['n_vehicles']} vehicles, {data['n_nodes']} nodes")
    
    # Run experiment
    run_experiment(data, local_args)
    
    return f"Completed processing {file_path}"

def run_experiment(data, args):
    cached_solution = {}

    # Create initial solution with all calls outsourced
    initial_solution = all_calls_outsorced(data)
    feasibility_check(initial_solution, data)
    initial_cost = cost_function(initial_solution, data)

    # Track best solutions and objectives
    best_overall_solution = initial_solution.copy()
    best_objectives = []
    total_time = 0
    feasible_count = 0
    best_overall_objective = initial_cost
    iteration_number_of_opt = []

    all_solutions = []

    func = final_alg
    
    print(f"\nRunning {func} simulated annealing and timestamp {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    for run in range(args.number_runs):
        print(f"\nRun {run+1}/{args.number_runs}")
        start_time = time.time()
        
        # Run algorithm
        solution, feasibility_tally, cached_solution, opt_iteration_number = func(initial_solution, data, seed=args.seed + run, cached_solution=cached_solution)
        
        # Evaluate solution
        feasible, reason = feasibility_check(solution, data)
        if feasible and len(solution) == len(best_overall_solution):
            cost = cost_function(solution, data)
            best_objectives.append(cost)
            feasible_count += feasibility_tally
            """total_operator_usage = [
                total_operator_usage[i] + operator_usage[i] for i in range(len(operator_usage))
            ]"""
            iteration_number_of_opt.append(opt_iteration_number)

            all_solutions.append(cost)
            if cost < best_overall_objective:
                best_overall_objective = cost
                best_overall_solution = solution.copy()
                best_run_index = run + 1
                
            print(f"Run {run+1} found feasible solution with cost: {cost:.2f}\n Time used: {time.time() - start_time:.2f} seconds")
        else:
            print(f"Run {run+1} found INfeasible solution with reason: {reason}. Correct length? {len(solution) == len(best_overall_solution)}")

        
        total_time += time.time() - start_time
    
    # Calculate improvement
    avg_objective = np.mean(best_objectives) if best_objectives else initial_cost
    improvement = 100.0 * (initial_cost - avg_objective) / initial_cost if initial_cost != 0 else 0
    
    # Print results
    print("\n===== Final Results =====")
    print(f"Best solution cost: {best_overall_objective:.2f}")
    print(f"Average solution cost: {avg_objective:.2f}")
    print(f"Improvement from initial solution: {improvement:.2f}%")
    print(f"Feasible solutions found: {feasible_count}")
    print(f"Total runtime: {total_time:.2f} seconds")
    # print the top cached solutions based on the number of times they were used    
    print(f"Top cached solutions: {sorted(cached_solution.items(), key=lambda x: x[1][2], reverse=True)[:1]}")
    print(f"All costs of all solutions: {all_solutions}")
    
    if not args.test and best_overall_solution is not None:
        result_file = args.file_path.replace(".txt", "_solution.txt")

        # Save results to file
        new_results = (f"Algorithm: Adaptive Weights Simulated Annealing.\n")
        new_results += (f"Seed: {args.seed}\n")
        new_results += (f"Best solution cost: {best_overall_objective:.2f}\n")
        new_results += (f"Best solution: {list(best_overall_solution)}\n")
        new_results += (f"Average cost: {avg_objective:.2f}\n")
        new_results += (f"Improvement: {improvement:.2f}%\n")
        new_results += (f"Feasible solutions: {feasible_count} for {10000*args.number_runs} passes\n")
        # new_results += (f"Operator usage: {total_operator_usage}\n")
        new_results += (f"Iteration numbers of optimal solutions of each run: {iteration_number_of_opt}\n")
        new_results += (f"All costs of all solutions: {all_solutions}")

        new_results += (f"Runtime: {total_time:.2f} seconds\n")
        new_results += f"Time of experiment: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        new_results += "------------------\n"

        # Append new results if they don't exist
        with open(result_file, 'a') as file:
                file.write(new_results)
    else:
        print('Test so no file written')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vehicle Routing Problem Solver')
    parser.add_argument('--file_path', type=str, default=file_paths[0], help='Path to the problem file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--number_runs', type=int, default=10, help='Number of runs')
    parser.add_argument('--test', action='store_true', help='Test mode, no output files')
    parser.add_argument('--full', action='store_true', help='Run with all dataset files')
    parser.add_argument('--parallel', action='store_true', help='Run problem instances in parallel')
    args = parser.parse_args()
    
    if args.full:
        if args.parallel:
            print(f"Running all instances in parallel with {multiprocessing.cpu_count()} available CPUs...")
            # Create a pool of worker processes
            with multiprocessing.Pool() as pool:
                # Map each file path to a worker process
                results = pool.starmap(
                    process_instance,
                    [(file_path, args) for file_path in file_paths]
                )
                
                # Print results when all processes are done
                for result in results:
                    print(result)
        else:
            for file_path in file_paths:
                args.file_path = file_path
                # Load problem data
                print(f"Loading problem from {args.file_path}...")
                data = load_problem(args.file_path)
                
                # Print problem statistics
                print(f"Problem size: {data['n_calls']} calls, {data['n_vehicles']} vehicles, {data['n_nodes']} nodes")
                
                # Run experiment
                run_experiment(data, args)
    else:
        # Load problem data
        print(f"Loading problem from {args.file_path}...")
        data = load_problem(args.file_path)
        
        # Print problem statistics
        print(f"Problem size: {data['n_calls']} calls, {data['n_vehicles']} vehicles, {data['n_nodes']} nodes")
        
        # Run experiment
        run_experiment(data, args)