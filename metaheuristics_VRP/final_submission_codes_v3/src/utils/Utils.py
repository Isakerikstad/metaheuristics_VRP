import numpy as np

def load_problem(filename):
    """

    :rtype: object
    :param filename: The address to the problem input file
    :return: named tuple object of the problem attributes
    """
    A = []
    B = []
    C = []
    D = []
    E = []
    with open(filename) as f:
        lines = f.readlines()
        num_nodes = int(lines[1])
        num_vehicles = int(lines[3])
        num_calls = int(lines[num_vehicles + 5 + 1])

        for i in range(num_vehicles):
            A.append(lines[1 + 4 + i].split(','))

        for i in range(num_vehicles):
            B.append(lines[1 + 7 + num_vehicles + i].split(','))

        for i in range(num_calls):
            C.append(lines[1 + 8 + num_vehicles * 2 + i].split(','))

        for j in range(num_nodes * num_nodes * num_vehicles):
            D.append(lines[1 + 2 * num_vehicles + num_calls + 9 + j].split(','))

        for i in range(num_vehicles * num_calls):
            E.append(lines[1 + 1 + 2 * num_vehicles + num_calls + 10 + j + i].split(','))
        f.close()

    Cargo = np.array(C, dtype=np.double)[:, 1:]
    D = np.array(D, dtype=int)

    TravelTime = np.zeros((num_vehicles + 1, num_nodes + 1, num_nodes + 1))
    TravelCost = np.zeros((num_vehicles + 1, num_nodes + 1, num_nodes + 1))
    for j in range(len(D)):
        TravelTime[D[j, 0]][D[j, 1], D[j, 2]] = D[j, 3]
        TravelCost[D[j, 0]][D[j, 1], D[j, 2]] = D[j, 4]

    VesselCapacity = np.zeros(num_vehicles)
    StartingTime = np.zeros(num_vehicles)
    FirstTravelTime = np.zeros((num_vehicles, num_nodes))
    FirstTravelCost = np.zeros((num_vehicles, num_nodes))
    A = np.array(A, dtype=int)
    for i in range(num_vehicles):
        VesselCapacity[i] = A[i, 3]
        StartingTime[i] = A[i, 2]
        for j in range(num_nodes):
            FirstTravelTime[i, j] = TravelTime[i + 1, A[i, 1], j + 1] + A[i, 2]
            FirstTravelCost[i, j] = TravelCost[i + 1, A[i, 1], j + 1]
    TravelTime = TravelTime[1:, 1:, 1:]
    TravelCost = TravelCost[1:, 1:, 1:]
    VesselCargo = np.zeros((num_vehicles, num_calls + 1))
    B = np.array(B, dtype=object)
    for i in range(num_vehicles):
        VesselCargo[i, np.array(B[i][1:], dtype=int)] = 1
    VesselCargo = VesselCargo[:, 1:]

    LoadingTime = np.zeros((num_vehicles + 1, num_calls + 1))
    UnloadingTime = np.zeros((num_vehicles + 1, num_calls + 1))
    PortCost = np.zeros((num_vehicles + 1, num_calls + 1))
    E = np.array(E, dtype=int)
    for i in range(num_vehicles * num_calls):
        LoadingTime[E[i, 0], E[i, 1]] = E[i, 2]
        UnloadingTime[E[i, 0], E[i, 1]] = E[i, 4]
        PortCost[E[i, 0], E[i, 1]] = E[i, 5] + E[i, 3]

    LoadingTime = LoadingTime[1:, 1:]
    UnloadingTime = UnloadingTime[1:, 1:]
    PortCost = PortCost[1:, 1:]
    output = {
        'n_nodes': num_nodes,
        'n_vehicles': num_vehicles,
        'n_calls': num_calls,
        'Cargo': Cargo,
        'TravelTime': TravelTime,
        'FirstTravelTime': FirstTravelTime,
        'VesselCapacity': VesselCapacity,
        'LoadingTime': LoadingTime,
        'UnloadingTime': UnloadingTime,
        'VesselCargo': VesselCargo,
        'TravelCost': TravelCost,
        'FirstTravelCost': FirstTravelCost,
        'PortCost': PortCost
    }
    return output

def feasibility_check(solution, problem):
    """

    :rtype: tuple
    :param solution: The input solution of order of calls for each vehicle to the problem
    :param problem: The pickup and delivery problem object
    :return: whether the problem is feasible and the reason for probable infeasibility
    """
    num_vehicles = problem['n_vehicles']
    Cargo = problem['Cargo']
    TravelTime = problem['TravelTime']
    FirstTravelTime = problem['FirstTravelTime']
    VesselCapacity = problem['VesselCapacity']
    LoadingTime = problem['LoadingTime']
    UnloadingTime = problem['UnloadingTime']
    VesselCargo = problem['VesselCargo']
    solution = np.append(solution, [0])
    ZeroIndex = np.array(np.where(solution == 0)[0], dtype=int)
    feasibility = True
    tempidx = 0
    c = 'Feasible'
    for i in range(num_vehicles):
        currentVPlan = solution[tempidx:ZeroIndex[i]]
        currentVPlan = currentVPlan - 1
        NoDoubleCallOnVehicle = len(currentVPlan)
        tempidx = ZeroIndex[i] + 1
        if NoDoubleCallOnVehicle > 0:

            if not np.all(VesselCargo[i, currentVPlan]):
                feasibility = False
                c = 'incompatible vessel and cargo'
                break
            else:
                LoadSize = 0
                currentTime = 0
                sortRout = np.sort(currentVPlan, kind='mergesort')
                I = np.argsort(currentVPlan, kind='mergesort')
                Indx = np.argsort(I, kind='mergesort')
                LoadSize -= Cargo[sortRout, 2]
                LoadSize[::2] = Cargo[sortRout[::2], 2]
                LoadSize = LoadSize[Indx]
                if np.any(VesselCapacity[i] - np.cumsum(LoadSize) < 0):
                    feasibility = False
                    c = 'Capacity exceeded'
                    break
                Timewindows = np.zeros((2, NoDoubleCallOnVehicle))
                Timewindows[0] = Cargo[sortRout, 6]
                Timewindows[0, ::2] = Cargo[sortRout[::2], 4]
                Timewindows[1] = Cargo[sortRout, 7]
                Timewindows[1, ::2] = Cargo[sortRout[::2], 5]

                Timewindows = Timewindows[:, Indx]

                PortIndex = Cargo[sortRout, 1].astype(int)
                PortIndex[::2] = Cargo[sortRout[::2], 0]
                PortIndex = PortIndex[Indx] - 1

                LU_Time = UnloadingTime[i, sortRout]
                LU_Time[::2] = LoadingTime[i, sortRout[::2]]
                LU_Time = LU_Time[Indx]
                Diag = TravelTime[i, PortIndex[:-1], PortIndex[1:]]
                FirstVisitTime = FirstTravelTime[i, int(Cargo[currentVPlan[0], 0] - 1)]

                RouteTravelTime = np.hstack((FirstVisitTime, Diag.flatten()))

                ArriveTime = np.zeros(NoDoubleCallOnVehicle)
                for j in range(NoDoubleCallOnVehicle):
                    ArriveTime[j] = np.max((currentTime + RouteTravelTime[j], Timewindows[0, j]))
                    if ArriveTime[j] > Timewindows[1, j]:
                        feasibility = False
                        c = 'Time window exceeded at call {}'.format(j)
                        break
                    currentTime = ArriveTime[j] + LU_Time[j]

    return feasibility, c

def cost_function(Solution, problem):
    """

    :param Solution: the proposed solution for the order of calls in each vehicle
    :param problem:
    :return:
    """

    num_vehicles = problem['n_vehicles']
    Cargo = problem['Cargo']
    TravelCost = problem['TravelCost']
    FirstTravelCost = problem['FirstTravelCost']
    PortCost = problem['PortCost']


    NotTransportCost = 0
    RouteTravelCost = np.zeros(num_vehicles)
    CostInPorts = np.zeros(num_vehicles)

    Solution = np.append(Solution, [0])
    ZeroIndex = np.array(np.where(Solution == 0)[0], dtype=int)
    tempidx = 0

    for i in range(num_vehicles + 1):
        currentVPlan = Solution[tempidx:ZeroIndex[i]]
        currentVPlan = currentVPlan - 1
        NoDoubleCallOnVehicle = len(currentVPlan)
        tempidx = ZeroIndex[i] + 1

        if i == num_vehicles:
            NotTransportCost = np.sum(Cargo[currentVPlan, 3]) / 2
        else:
            if NoDoubleCallOnVehicle > 0:
                sortRout = np.sort(currentVPlan, kind='mergesort')
                I = np.argsort(currentVPlan, kind='mergesort')
                Indx = np.argsort(I, kind='mergesort')

                PortIndex = Cargo[sortRout, 1].astype(int)
                PortIndex[::2] = Cargo[sortRout[::2], 0]
                PortIndex = PortIndex[Indx] - 1

                Diag = TravelCost[i, PortIndex[:-1], PortIndex[1:]]

                FirstVisitCost = FirstTravelCost[i, int(Cargo[currentVPlan[0], 0] - 1)]
                RouteTravelCost[i] = np.sum(np.hstack((FirstVisitCost, Diag.flatten())))
                CostInPorts[i] = np.sum(PortCost[i, currentVPlan]) / 2

    TotalCost = NotTransportCost + sum(RouteTravelCost) + sum(CostInPorts)
    return TotalCost

def all_calls_outsorced(data):
    """Create initial solution with all calls outsourced to dummy vehicle"""
    # Create a solution with all calls assigned to the dummy vehicle
    n_calls = data['n_calls']
    n_vehicles = data['n_vehicles']
    
    # One position for each pickup and delivery, plus n_vehicles zeros as separators
    solution = np.zeros(n_calls * 2 + n_vehicles, dtype=int)
    
    # Set positions for vehicle separators (0s)
    for i in range(n_vehicles):
        solution[i] = 0
    
    # Fill remaining positions with outsourced calls
    calls = np.repeat(np.arange(1, n_calls + 1), 2)
    solution[n_vehicles:] = calls

    # make into list
    solution = solution.tolist()
   
    return solution

def local_cost_function(route, data, vehicle_idx):
    """
    Calculate travel cost only for a vehicle's route.
    
    Args:
        route: List of calls in the route
        data: Problem data
        vehicle_idx: Vehicle index
    
    Returns:
        Total travel cost for the route
    """
    if not route:
        return 0
    
    # Get cargo data for port information
    cargo = data['Cargo']
    
    # Initialize travel cost
    total_cost = 0
    
    # Add port costs (loading/unloading)
    port_cost = 0
    calls_set = set()
    for call in route:
        if call not in calls_set:
            # This is a pickup
            port_cost += data['PortCost'][vehicle_idx, call-1] / 2
            calls_set.add(call)
        else:
            # This is a delivery
            port_cost += data['PortCost'][vehicle_idx, call-1] / 2
            calls_set.remove(call)
    
    # Add transport costs
    if len(route) > 0:
        # Add first leg cost (from vehicle start to first pickup)
        first_node = int(cargo[route[0]-1, 0]) - 1  # Pickup node of first call
        total_cost += data['FirstTravelCost'][vehicle_idx, first_node]
        
        # Track visited calls to determine if a position is pickup or delivery
        visited_calls = set()
        curr_node = first_node
        
        for i in range(1, len(route)):
            call_idx = route[i] - 1
            
            if route[i] in visited_calls:
                # This is a delivery - destination node
                visited_calls.remove(route[i])
                next_node = int(cargo[call_idx, 1]) - 1
            else:
                # This is a pickup - origin node
                visited_calls.add(route[i])
                next_node = int(cargo[call_idx, 0]) - 1
            
            # Add travel cost between nodes
            total_cost += data['TravelCost'][vehicle_idx, curr_node, next_node]
            curr_node = next_node
    
    # Return total cost (transport + port costs)
    return total_cost + port_cost

def vehicle_feasibility_check(solution, problem, vehicle_idx):
    """
    Checks the feasibility of a single vehicle's route.
    
    Args:
        solution: The solution array containing all routes
        problem: The problem data dictionary
        vehicle_idx: Index of the vehicle to check
    
    Returns:
        bool: True if the vehicle's route is feasible, False otherwise
    """
    # Extract problem data
    Cargo = problem['Cargo']
    TravelTime = problem['TravelTime']
    FirstTravelTime = problem['FirstTravelTime']
    VesselCapacity = problem['VesselCapacity']
    LoadingTime = problem['LoadingTime']
    UnloadingTime = problem['UnloadingTime']
    VesselCargo = problem['VesselCargo']
    
    # Find this vehicle's route
    if isinstance(solution, np.ndarray):
        solution = solution.tolist()
        
    vehicle_bounds = [i for i, x in enumerate(solution) if x == 0]
    
    # Determine start and end indices for this vehicle's route
    start_idx = 0 if vehicle_idx == 0 else vehicle_bounds[vehicle_idx - 1] + 1 if vehicle_idx - 1 < len(vehicle_bounds) else len(solution)
    end_idx = vehicle_bounds[vehicle_idx] if vehicle_idx < len(vehicle_bounds) else len(solution)
    
    # Extract current vehicle's plan
    currentVPlan = solution[start_idx:end_idx]
    
    # Adjust call indices (subtract 1 for zero-indexing in the data)
    currentVPlan = [c - 1 for c in currentVPlan if c > 0]  # Remove vehicle separators and adjust indices
    
    # If route is empty, it's trivially feasible
    NoDoubleCallOnVehicle = len(currentVPlan)
    if NoDoubleCallOnVehicle == 0:
        return True, 'Feasible'
    
    # Check vehicle-cargo compatibility
    if not np.all(VesselCargo[vehicle_idx, currentVPlan]):
        return False, 'incompatible_vessel_cargo'
    
    # Check capacity constraints
    LoadSize = 0
    currentTime = 0
    sortRout = np.sort(currentVPlan, kind='mergesort')
    I = np.argsort(currentVPlan, kind='mergesort')
    Indx = np.argsort(I, kind='mergesort')
    
    # Calculate load sizes
    LoadSize = np.zeros(NoDoubleCallOnVehicle)
    LoadSize -= Cargo[sortRout, 2]
    LoadSize[::2] = Cargo[sortRout[::2], 2]
    LoadSize = LoadSize[Indx]
    
    # Check if capacity is exceeded at any point
    if np.any(VesselCapacity[vehicle_idx] - np.cumsum(LoadSize) < 0):
        return False, 'vehicle_overloaded'
    
    # Check time window constraints
    Timewindows = np.zeros((2, NoDoubleCallOnVehicle))
    Timewindows[0] = Cargo[sortRout, 6]
    Timewindows[0, ::2] = Cargo[sortRout[::2], 4]
    Timewindows[1] = Cargo[sortRout, 7]
    Timewindows[1, ::2] = Cargo[sortRout[::2], 5]
    
    Timewindows = Timewindows[:, Indx]
    
    # Get port indices
    PortIndex = Cargo[sortRout, 1].astype(int)
    PortIndex[::2] = Cargo[sortRout[::2], 0]
    PortIndex = PortIndex[Indx] - 1  # Convert to 0-indexed
    
    # Calculate loading/unloading times
    LU_Time = UnloadingTime[vehicle_idx, sortRout]
    LU_Time[::2] = LoadingTime[vehicle_idx, sortRout[::2]]
    LU_Time = LU_Time[Indx]
    
    # Calculate route travel times
    Diag = TravelTime[vehicle_idx, PortIndex[:-1], PortIndex[1:]]
    FirstVisitTime = FirstTravelTime[vehicle_idx, int(Cargo[currentVPlan[0], 0] - 1)]
    RouteTravelTime = np.hstack((FirstVisitTime, Diag.flatten()))
    
    # Check time windows
    ArriveTime = np.zeros(NoDoubleCallOnVehicle)
    for j in range(NoDoubleCallOnVehicle):
        ArriveTime[j] = np.max((currentTime + RouteTravelTime[j], Timewindows[0, j]))
        if ArriveTime[j] > Timewindows[1, j]:
            return False, 'time_window_violation'
        currentTime = ArriveTime[j] + LU_Time[j]
    
    # All constraints satisfied
    return True, 'Feasible'

def cost_transport_only(route, data, vehicle_idx):
    """
    Calculate travel cost only for a vehicle's route.
    
    Args:
        route: List of calls in the route
        data: Problem data
        vehicle_idx: Vehicle index
    
    Returns:
        Total travel cost for the route
    """
    if not route:
        return 0
    
    # Convert route to numpy array for processing
    route_array = np.array(route)
    
    # Get cargo data for port information
    cargo = data['Cargo']
    
    # Get first node (pickup) of first call
    first_node = int(cargo[route_array[0]-1, 0]) - 1
    
    # Calculate first leg cost
    total_cost = data['FirstTravelCost'][vehicle_idx, first_node]
    
    # Calculate cost between each pair of calls
    for i in range(len(route_array) - 1):
        # Determine if current and next positions are pickups or deliveries
        curr_call = route_array[i] - 1
        next_call = route_array[i+1] - 1
        
        # Get origin/destination nodes
        curr_node = int(cargo[curr_call, 1]) - 1  # Default to delivery
        next_node = int(cargo[next_call, 0]) - 1  # Default to pickup
        
        # If same call appears twice in sequence, handle it differently
        if curr_call == next_call:
            curr_node = int(cargo[curr_call, 0]) - 1  # Origin for pickup
            next_node = int(cargo[next_call, 1]) - 1  # Destination for delivery
        
        # Add travel cost between these nodes
        total_cost += data['TravelCost'][vehicle_idx, curr_node, next_node]
    
    return total_cost
