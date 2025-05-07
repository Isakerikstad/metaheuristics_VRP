"""
Operators for use in VRP solution algorithms.
"""
from src.operators.greedy_insertion import expensive_greedy_insertion
from src.operators.shaw_removal import shaw_removal, shaw_removal_clipped
from src.operators.random_removal import random_removal, random_removal_clipped
from src.operators.remove_highest_cost_call import remove_highest_cost_call, remove_highest_cost_call_clipped
from src.operators.greedy_insertion import regret_k_insertion
from src.operators.greedy_insertion import regret_k_insertion_3
from src.operators.greedy_insertion import regret_k_insertion_4
from src.operators.vehicle_swap import vehicle_swap

from src.operators.greedy_insertion import greedy_timing
