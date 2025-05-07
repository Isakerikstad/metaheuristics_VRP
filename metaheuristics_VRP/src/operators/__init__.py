"""
Operators for use in VRP solution algorithms.
"""

from src.operators.basic_inserter import basic_inserter, basic_inserter_v2, basic_inserter_v4
from src.operators.greedy_insertion import expensive_greedy_insertion
from src.operators.insert_legally import insert_legally

from src.operators.shaw_removal import shaw_removal, shaw_removal_clipped
from src.operators.random_removal import random_removal, random_removal_clipped
from src.operators.dummy_removal import dummy_removal, dummy_removal_clipped
from src.operators.remove_highest_cost_call import remove_highest_cost_call, remove_highest_cost_call_clipped
from src.operators.greedy_insertion import regret_k_insertion
from src.operators.greedy_insertion import regret_k_insertion_3
from src.operators.greedy_insertion import regret_k_insertion_4

from src.operators.vehicle_swap import vehicle_swap

__all__ = [basic_inserter, basic_inserter_v2, basic_inserter_v4, expensive_greedy_insertion, insert_legally, shaw_removal, shaw_removal_clipped, random_removal, random_removal_clipped, dummy_removal, dummy_removal_clipped, remove_highest_cost_call, remove_highest_cost_call_clipped, regret_k_insertion, regret_k_insertion_3, regret_k_insertion_4]

from src.operators.greedy_insertion import greedy_timing
