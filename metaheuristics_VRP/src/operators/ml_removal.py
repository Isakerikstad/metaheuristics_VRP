import numpy as np
import pandas as pd
import pickle
import os
from copy import deepcopy
import itertools
import random
from collections import defaultdict

# Try importing AutoGluon - wrap in try/except in case it's not installed
try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    print("AutoGluon not available. Install with: pip install autogluon")

# Constants
ML_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
TRAINING_DATA_PATH = os.path.join(ML_MODEL_PATH, 'training_data.csv')
MODEL_PATH = os.path.join(ML_MODEL_PATH, 'removal_model')
RETRAIN_MODEL_PATH = os.path.join(ML_MODEL_PATH, 'removal_model_retrained')
GROUP_MODEL_PATH = os.path.join(ML_MODEL_PATH, 'group_removal_model')

class MLRemovalOperator:
    """
    A machine learning-based removal operator that learns which calls to remove
    based on historic performance data.
    """
    
    def __init__(self, training_mode=True, training_iterations=5000, retrain_at=9000):
        """
        Initialize the ML removal operator.
        
        Args:
            training_mode: If True, collect data for training. If False, use trained model.
            training_iterations: Number of iterations to collect before training the model
            retrain_at: Iteration at which to perform retraining
        """
        self.training_mode = training_mode
        self.training_iterations = training_iterations
        self.retrain_at = retrain_at
        self.iterations_seen = 0
        self.training_data = []
        self.group_training_data = []  # For training on groups of calls
        self.model = None
        self.group_model = None  # Model for predicting improvement from groups of calls
        self.call_features = {}  # Cache for call features
        self.current_iteration = 0
        self.retraining_triggered = False
        
        # Exploration parameters (higher = more random exploration)
        self.exploration_rate = 0.7  # Start with high exploration
        self.min_exploration_rate = 0.3  # Never go below this exploration rate
        self.exploration_decay = 0.9995  # Slowly decay exploration rate
        
        # Group removal settings
        self.enable_group_prediction = True  # Whether to use group-based prediction
        self.max_group_size = 3  # Maximum size of groups to consider
        
        # Call relationship cache
        self.call_relationships = defaultdict(dict)  # Cache for call relationship features
        
        # Create models directory if it doesn't exist
        os.makedirs(ML_MODEL_PATH, exist_ok=True)
        
        # Load model if in prediction mode
        if not training_mode and AUTOGLUON_AVAILABLE:
            self._load_model()
            self._load_group_model()
    
    def _load_model(self, retrained=False):
        """Load the trained model if available."""
        try:
            model_path = RETRAIN_MODEL_PATH if retrained else MODEL_PATH
            if os.path.exists(model_path):
                self.model = TabularPredictor.load(model_path)
                print(f"Loaded ML removal model from {model_path}")
                return True
            else:
                print(f"No trained model found at {model_path}")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _load_group_model(self):
        """Load the group prediction model if available."""
        try:
            if os.path.exists(GROUP_MODEL_PATH):
                self.group_model = TabularPredictor.load(GROUP_MODEL_PATH)
                print(f"Loaded group removal model from {GROUP_MODEL_PATH}")
                return True
            else:
                print(f"No group model found at {GROUP_MODEL_PATH}")
                return False
        except Exception as e:
            print(f"Error loading group model: {e}")
            return False
    
    def _extract_call_features(self, call, data):
        """Extract features for a specific call from the problem data."""
        # Check if we've already computed features for this call
        if call in self.call_features:
            return self.call_features[call]
        
        # Extract static features for the call
        call_idx = call - 1  # Convert to 0-index
        
        # Get origin and destination from Cargo array
        # Cargo array format: [origin_node, destination_node, size, cost_of_not_transporting, pickup_time_windows, delivery_time_windows]
        features = {
            'call_id': call,
            'origin': int(data['Cargo'][call_idx][0]),
            'destination': int(data['Cargo'][call_idx][1]),
            'size': data['Cargo'][call_idx][2],
            'compatible_vehicles': sum(data['VesselCargo'][:, call_idx]),
            'outsource_cost': data['Cargo'][call_idx][3],  # Cost of not transporting
        }
        
        # Time window features - get from Cargo array
        features['pickup_earliest'] = data['Cargo'][call_idx][4]
        features['pickup_latest'] = data['Cargo'][call_idx][5]
        features['delivery_earliest'] = data['Cargo'][call_idx][6]
        features['delivery_latest'] = data['Cargo'][call_idx][7]
        features['time_window_width'] = features['delivery_latest'] - features['pickup_earliest']
        features['time_window_tightness'] = (features['delivery_latest'] - features['delivery_earliest']) / max(1, (features['pickup_latest'] - features['pickup_earliest']))
        
        # Cache the features
        self.call_features[call] = features
        return features

    def _extract_solution_features(self, solution, data):
        """Extract features about the current solution state."""
        # Find vehicle boundaries
        vehicle_boundaries = [i for i, x in enumerate(solution) if x == 0]
        
        # Identify calls in each vehicle and the dummy
        vehicle_calls = {}
        for v in range(data['n_vehicles']):
            start_idx = 0 if v == 0 else vehicle_boundaries[v-1] + 1
            end_idx = vehicle_boundaries[v] if v < len(vehicle_boundaries) else len(solution)
            route = solution[start_idx:end_idx]
            vehicle_calls[v] = [x for x in route if x > 0]
        
        # Dummy vehicle calls (those after the last vehicle marker)
        if len(vehicle_boundaries) > 0:
            dummy_calls = solution[vehicle_boundaries[-1]+1:]
            dummy_calls = [x for x in dummy_calls if x > 0]
        else:
            dummy_calls = []
        
        # Calculate how many calls are in each vehicle
        vehicle_load = {v: len(set(calls)) for v, calls in vehicle_calls.items()}
        
        # Calculate average travel cost for each vehicle
        vehicle_costs = {}
        for v in range(data['n_vehicles']):
            if v in vehicle_calls and vehicle_calls[v]:
                # Extract unique calls in this vehicle
                unique_calls = list(set(vehicle_calls[v]))
                if unique_calls:
                    # Calculate average travel cost
                    vehicle_costs[v] = self._estimate_vehicle_cost(unique_calls, data, v)
                else:
                    vehicle_costs[v] = 0
            else:
                vehicle_costs[v] = 0
        
        return {
            'vehicle_calls': vehicle_calls,
            'dummy_calls': dummy_calls,
            'n_outsourced': len(set(dummy_calls)) // 2,  # Count unique calls
            'vehicle_load': vehicle_load,
            'vehicle_costs': vehicle_costs
        }
    
    def _estimate_vehicle_cost(self, calls, data, vehicle_idx):
        """Estimate the cost of a vehicle's route based on its calls."""
        if not calls:
            return 0
            
        # Simple estimate based on travel between nodes
        total_cost = 0
        cargo = data['Cargo']
        
        # This is just a rough estimate - we don't have the actual route
        if len(calls) >= 2:
            # For each call pair, calculate the travel cost between them
            origin_nodes = [int(cargo[c-1][0]) - 1 for c in calls]
            dest_nodes = [int(cargo[c-1][1]) - 1 for c in calls]
            
            # Calculate average travel cost between nodes
            total_pairs = 0
            for i in range(len(origin_nodes)):
                for j in range(i+1, len(origin_nodes)):
                    # Cost from i's origin to j's origin
                    if vehicle_idx < data['TravelCost'].shape[0]:
                        cost_oo = data['TravelCost'][vehicle_idx, origin_nodes[i], origin_nodes[j]]
                        # Cost from i's destination to j's destination
                        cost_dd = data['TravelCost'][vehicle_idx, dest_nodes[i], dest_nodes[j]]
                        # Cost from i's origin to j's destination
                        cost_od = data['TravelCost'][vehicle_idx, origin_nodes[i], dest_nodes[j]]
                        # Cost from i's destination to j's origin
                        cost_do = data['TravelCost'][vehicle_idx, dest_nodes[i], origin_nodes[j]]
                        
                        # Average of these travel costs
                        total_cost += (cost_oo + cost_dd + cost_od + cost_do) / 4
                        total_pairs += 1
            
            if total_pairs > 0:
                return total_cost / total_pairs
        
        return 0
    
    def _calculate_call_relationships(self, call1, call2, data):
        """Calculate relationship features between two calls."""
        # Check cache first
        if call2 in self.call_relationships[call1]:
            return self.call_relationships[call1][call2]
        
        call1_idx = call1 - 1
        call2_idx = call2 - 1
        cargo = data['Cargo']
        
        # Origin and destination nodes for both calls
        origin1 = int(cargo[call1_idx][0]) - 1
        dest1 = int(cargo[call1_idx][1]) - 1
        origin2 = int(cargo[call2_idx][0]) - 1
        dest2 = int(cargo[call2_idx][1]) - 1
        
        # Calculate travel time compatibility across all vehicles
        travel_costs = []
        for v in range(min(data['n_vehicles'], data['TravelCost'].shape[0])):
            # Check if both calls are compatible with this vehicle
            if (data['VesselCargo'][v, call1_idx] == 1 and 
                data['VesselCargo'][v, call2_idx] == 1):
                # Calculate costs between origin-origin, dest-dest, etc.
                cost_oo = data['TravelCost'][v, origin1, origin2]
                cost_dd = data['TravelCost'][v, dest1, dest2]
                cost_od = data['TravelCost'][v, origin1, dest2]
                cost_do = data['TravelCost'][v, dest1, origin2]
                
                # Minimum cost path between these calls
                min_cost = min(cost_oo, cost_dd, cost_od, cost_do)
                travel_costs.append(min_cost)
        
        # If no compatible vehicles, use a high value
        avg_travel_cost = np.mean(travel_costs) if travel_costs else float('inf')
        
        # Time window compatibility
        pickup_latest1 = cargo[call1_idx][5]
        pickup_latest2 = cargo[call2_idx][5]
        delivery_earliest1 = cargo[call1_idx][6]
        delivery_earliest2 = cargo[call2_idx][6]
        
        # Check if call1 can be done before call2
        c1_before_c2 = pickup_latest1 < delivery_earliest2
        # Check if call2 can be done before call1
        c2_before_c1 = pickup_latest2 < delivery_earliest1
        
        # Size compatibility (are they similar in size?)
        size_ratio = min(cargo[call1_idx][2], cargo[call2_idx][2]) / max(1, max(cargo[call1_idx][2], cargo[call2_idx][2]))
        
        # Create relationship features
        relationship = {
            'avg_travel_cost': float(avg_travel_cost),
            'c1_before_c2': int(c1_before_c2),
            'c2_before_c1': int(c2_before_c1),
            'size_ratio': float(size_ratio),
        }
        
        # Cache for future use
        self.call_relationships[call1][call2] = relationship
        self.call_relationships[call2][call1] = relationship
        
        return relationship
    
    def _get_call_features_in_solution(self, call, solution, data):
        """Get combined features for a call in the current solution context."""
        # Get basic call features
        features = self._extract_call_features(call, data)
        
        # Find where the call is currently placed
        if isinstance(solution, np.ndarray):
            solution = solution.tolist()
        
        # Find vehicle boundaries
        vehicle_boundaries = [i for i, x in enumerate(solution) if x == 0]
        
        # Find positions of the call in solution
        call_positions = [i for i, x in enumerate(solution) if x == call]
        if len(call_positions) >= 2:
            pickup_idx = min(call_positions)
            delivery_idx = max(call_positions)
            
            # Determine which vehicle contains the call
            in_vehicle = -1  # -1 means dummy vehicle
            for v in range(len(vehicle_boundaries)):
                if v == 0:
                    if pickup_idx < vehicle_boundaries[0]:
                        in_vehicle = 0
                        break
                else:
                    if vehicle_boundaries[v-1] < pickup_idx < vehicle_boundaries[v]:
                        in_vehicle = v
                        break
            
            features['in_vehicle'] = in_vehicle
            features['is_outsourced'] = 1 if in_vehicle == -1 else 0
            
            # Get route position features
            if in_vehicle >= 0:
                start_idx = 0 if in_vehicle == 0 else vehicle_boundaries[in_vehicle-1] + 1
                end_idx = vehicle_boundaries[in_vehicle]
                route = solution[start_idx:end_idx]
                features['route_length'] = len([x for x in route if x > 0]) // 2  # Count unique calls
                features['route_position'] = (pickup_idx - start_idx) / max(1, len(route))
                
                # Get nearby calls in the same route (calls that are adjacent to this one)
                route_calls = [x for x in route if x > 0 and x != call]
                unique_route_calls = list(set(route_calls))
                
                # If there are other calls in this vehicle, calculate relationship features
                if unique_route_calls:
                    # Find closest call in the route (by position)
                    call_positions_in_route = [i for i, x in enumerate(route) if x == call]
                    other_positions = [(i, x) for i, x in enumerate(route) if x > 0 and x != call]
                    
                    if call_positions_in_route and other_positions:
                        # Find closest call by position
                        closest_pos_diff = float('inf')
                        closest_call = None
                        
                        for call_pos in call_positions_in_route:
                            for other_pos, other_call in other_positions:
                                pos_diff = abs(call_pos - other_pos)
                                if pos_diff < closest_pos_diff:
                                    closest_pos_diff = pos_diff
                                    closest_call = other_call
                        
                        if closest_call is not None:
                            # Get relationship features with closest call
                            relationship = self._calculate_call_relationships(call, closest_call, data)
                            features['closest_call'] = closest_call
                            features['closest_call_travel_cost'] = relationship['avg_travel_cost']
                            features['closest_call_time_compatible'] = max(relationship['c1_before_c2'], relationship['c2_before_c1'])
                            features['closest_call_size_ratio'] = relationship['size_ratio']
                
                # Calculate vehicle utilization
                if in_vehicle < data['VesselCapacity'].shape[0]:
                    # Get all calls in this vehicle
                    vehicle_calls = list(set([x for x in solution[start_idx:end_idx] if x > 0]))
                    total_size = 0
                    for c in vehicle_calls:
                        c_idx = c - 1
                        if c_idx < len(data['Cargo']):
                            total_size += data['Cargo'][c_idx][2]
                    
                    capacity = data['VesselCapacity'][in_vehicle]
                    features['vehicle_utilization'] = total_size / max(1, capacity)
        else:
            # Call not in solution or data inconsistency
            features['in_vehicle'] = -2  # Error state
            features['is_outsourced'] = -1
            features['route_length'] = 0
            features['route_position'] = -1
            features['vehicle_utilization'] = 0
            
        return features
    
    def _get_group_features(self, calls, solution, data):
        """Extract features for a group of calls in the current solution."""
        features = {}
        
        # Get features for each individual call
        call_features = [self._get_call_features_in_solution(call, solution, data) for call in calls]
        
        # Basic group statistics
        features['group_size'] = len(calls)
        features['avg_size'] = np.mean([f['size'] for f in call_features])
        features['total_size'] = sum([f['size'] for f in call_features])
        features['max_size'] = max([f['size'] for f in call_features])
        features['avg_outsource_cost'] = np.mean([f.get('outsource_cost', 0) for f in call_features])
        features['total_outsource_cost'] = sum([f.get('outsource_cost', 0) for f in call_features])
        
        # Vehicle distribution
        vehicles = [f['in_vehicle'] for f in call_features if f['in_vehicle'] >= 0]
        features['unique_vehicles'] = len(set(vehicles))
        features['all_same_vehicle'] = 1 if len(set(vehicles)) <= 1 else 0
        features['all_outsourced'] = 1 if all(f['is_outsourced'] == 1 for f in call_features) else 0
        
        # Call relationship features within the group
        if len(calls) >= 2:
            relationship_features = []
            for idx1, call1 in enumerate(calls):
                for idx2 in range(idx1 + 1, len(calls)):
                    call2 = calls[idx2]
                    rel = self._calculate_call_relationships(call1, call2, data)
                    relationship_features.append(rel)
            
            if relationship_features:
                features['avg_travel_cost'] = np.mean([r['avg_travel_cost'] for r in relationship_features])
                features['min_travel_cost'] = min([r['avg_travel_cost'] for r in relationship_features])
                features['max_travel_cost'] = max([r['avg_travel_cost'] for r in relationship_features])
                features['avg_size_ratio'] = np.mean([r['size_ratio'] for r in relationship_features])
                features['time_compatible'] = sum([max(r['c1_before_c2'], r['c2_before_c1']) for r in relationship_features]) / len(relationship_features)
        
        return features
    
    def record_result(self, removed_calls, initial_solution, new_solution, improvement, data, iteration=None):
        """
        Record the result of removing and reinserting calls.
        
        Args:
            removed_calls: List of calls that were removed
            initial_solution: Solution before removal
            new_solution: Solution after reinsertion
            improvement: Cost improvement (negative means better)
            data: Problem data dictionary
            iteration: Current iteration number
        """
        # Update current iteration if provided
        if iteration is not None:
            self.current_iteration = iteration
            
        # Update iteration counter
        self.iterations_seen += 1
        
        # In training mode or retraining phase, collect data
        should_collect = self.training_mode or (not self.training_mode and self.retraining_triggered)
        if should_collect:
            # Extract solution features
            solution_features = self._extract_solution_features(initial_solution, data)
            
            # For each removed call, create a training record
            for call in removed_calls:
                try:
                    call_features = self._get_call_features_in_solution(call, initial_solution, data)
                    
                    # Ensure all features are scalar values, not lists or arrays
                    scalar_features = {}
                    for key, value in call_features.items():
                        if isinstance(value, (list, np.ndarray)):
                            # For lists/arrays, we'll use the first element or mean
                            try:
                                if len(value) > 0:
                                    scalar_features[key] = float(value[0])  # Take first element
                                else:
                                    scalar_features[key] = 0.0  # Default for empty arrays
                            except (TypeError, IndexError):
                                scalar_features[key] = float(np.mean(value))  # Use mean for arrays
                        else:
                            # Try to convert to float for consistency
                            try:
                                scalar_features[key] = float(value)
                            except (ValueError, TypeError):
                                scalar_features[key] = str(value)  # Use string if can't convert to float
                    
                    # Create combined features
                    record = {
                        'call_id': int(call),
                        'improvement': 1 if improvement < 0 else 0,  # Binary classification feature
                        'improvement_value': float(-improvement),  # Make positive for better ML target
                        'iteration': int(self.current_iteration),
                        **scalar_features
                    }
                    
                    # Add to training data - all data goes to the same list now
                    self.training_data.append(record)
                except Exception as e:
                    print(f"Error recording data for call {call}: {e}")
                    
            # Also collect group data if we have multiple calls removed and group prediction is enabled
            if len(removed_calls) >= 2 and self.enable_group_prediction:
                try:
                    # Get group features
                    group_features = self._get_group_features(removed_calls, initial_solution, data)
                    
                    # Create group record
                    group_record = {
                        'group_size': len(removed_calls),
                        'calls': str(sorted(removed_calls)),  # Convert to string for storage
                        'improvement': 1 if improvement < 0 else 0,
                        'improvement_value': float(-improvement),
                        'iteration': int(self.current_iteration),
                    }
                    
                    # Add scalar group features
                    for key, value in group_features.items():
                        if not isinstance(value, (list, dict, np.ndarray)):
                            try:
                                group_record[key] = float(value)
                            except (ValueError, TypeError):
                                group_record[key] = str(value)
                    
                    # Add to group training data - same list for initial training and retraining
                    self.group_training_data.append(group_record)
                except Exception as e:
                    print(f"Error recording group data: {e}")
        
        # Check if initial training is complete
        if self.training_mode and self.iterations_seen >= self.training_iterations:
            self._train_model()
            if self.enable_group_prediction and len(self.group_training_data) >= 100:
                self._train_group_model()
            self.training_mode = False  # Switch to prediction mode
            return
                
        # If we're in prediction mode and not yet retraining, check if it's time to retrain
        if not self.training_mode and not self.retraining_triggered:
            # If we hit the retraining iteration, start collecting data again
            if self.current_iteration >= self.retrain_at:
                print(f"Iteration {self.current_iteration}: Beginning collection of retraining data")
                self.retraining_triggered = True
            return
                
        # If we're in retraining phase and collected enough samples, retrain the model
        if self.retraining_triggered and len(self.training_data) > self.iterations_seen + 1000:
            print(f"Collected {len(self.training_data)} total samples for retraining")
            self._retrain_model()
            if self.enable_group_prediction and len(self.group_training_data) >= 200:
                self._train_group_model()
            self.retraining_triggered = False  # Reset retraining flag
    
    def _train_model(self):
        """Train the ML model using collected data."""
        if not AUTOGLUON_AVAILABLE:
            print("AutoGluon not available. Cannot train model.")
            return False
            
        if len(self.training_data) < 100:
            print("Not enough training data to train model.")
            return False
            
        # Convert to DataFrame
        df = pd.DataFrame(self.training_data)
        
        # Save training data
        df.to_csv(TRAINING_DATA_PATH, index=False)
        print(f"Saved {len(df)} training samples to {TRAINING_DATA_PATH}")
        
        # Train model
        print("Training removal operator ML model...")
        try:
            # Use AutoGluon to train a model predicting improvement
            predictor = TabularPredictor(
                label='improvement_value',
                path=MODEL_PATH,
                eval_metric='root_mean_squared_error'
            )
            predictor.fit(
                train_data=df,
                time_limit=120,  # 2 minutes time limit
                presets='medium_quality'
            )
            self.model = predictor
            print(f"Model trained and saved to {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def _train_group_model(self):
        """Train a model for group predictions."""
        if not AUTOGLUON_AVAILABLE:
            print("AutoGluon not available. Cannot train group model.")
            return False
            
        if len(self.group_training_data) < 100:
            print("Not enough group training data to train model.")
            return False
            
        # Convert to DataFrame
        df = pd.DataFrame(self.group_training_data)
        
        # Save training data
        group_data_path = os.path.join(ML_MODEL_PATH, 'group_training_data.csv')
        df.to_csv(group_data_path, index=False)
        print(f"Saved {len(df)} group training samples to {group_data_path}")
        
        # Train model
        print("Training group removal operator ML model...")
        try:
            # Use AutoGluon to train a model predicting improvement for groups
            predictor = TabularPredictor(
                label='improvement_value',
                path=GROUP_MODEL_PATH,
                eval_metric=None
            )
            predictor.fit(
                train_data=df,
                time_limit=120,  # 2 minutes time limit
                presets='medium_quality'
            )
            self.group_model = predictor
            print(f"Group model trained and saved to {GROUP_MODEL_PATH}")
            return True
        except Exception as e:
            print(f"Error training group model: {e}")
            return False
    
    def _retrain_model(self):
        """Retrain the ML model using all collected data."""
        if not AUTOGLUON_AVAILABLE:
            print("AutoGluon not available. Cannot retrain model.")
            return False
            
        if len(self.training_data) < 100:
            print("Not enough retraining data. Skipping retraining.")
            return False
            
        print(f"Retraining model with {len(self.training_data)} total samples...")
        
        try:
            # Use all collected data for retraining
            df = pd.DataFrame(self.training_data)
            
            # Save combined training data
            retrain_data_path = os.path.join(ML_MODEL_PATH, 'retraining_data.csv')
            df.to_csv(retrain_data_path, index=False)
            print(f"Saved {len(df)} combined training samples to {retrain_data_path}")
            
            # Train new model
            predictor = TabularPredictor(
                label='improvement_value',
                path=RETRAIN_MODEL_PATH,
                eval_metric=None # Keep open so AutoGluon can choose the best metric
            )
            predictor.fit(
                train_data=df,
                time_limit=180,  # 3 minutes for retraining
                presets='high_quality'  # Higher quality for retraining
            )
            
            # Switch to the retrained model
            self.model = predictor
            print(f"Model retrained and saved to {RETRAIN_MODEL_PATH}")
            
            return True
        except Exception as e:
            print(f"Error retraining model: {e}")
            return False
    
    def _select_calls_by_ml(self, solution, data, n_calls):
        """Select calls to remove based on ML predictions."""
        if not AUTOGLUON_AVAILABLE or self.model is None:
            return None
        
        # Get all unique calls in the solution
        calls = set(int(x) for x in solution if x > 0)
        if not calls:
            return []
            
        # Create feature dataframe for all calls
        features_list = []
        for call in calls:
            try:
                features = self._get_call_features_in_solution(call, solution, data)
                # Only include features that are present in all calls to avoid DataFrame issues
                basic_features = {
                    'call_id': call,
                    'origin': features.get('origin', 0),
                    'destination': features.get('destination', 0),
                    'size': features.get('size', 0),
                    'compatible_vehicles': features.get('compatible_vehicles', 0),
                    'outsource_cost': features.get('outsource_cost', 0),
                    'pickup_earliest': features.get('pickup_earliest', 0),
                    'pickup_latest': features.get('pickup_latest', 0),
                    'delivery_earliest': features.get('delivery_earliest', 0), 
                    'delivery_latest': features.get('delivery_latest', 0),
                    'time_window_width': features.get('time_window_width', 0),
                    'time_window_tightness': features.get('time_window_tightness', 1.0),
                    'in_vehicle': features.get('in_vehicle', -1),
                    'is_outsourced': features.get('is_outsourced', 0),
                    'route_length': features.get('route_length', 0),
                    'route_position': features.get('route_position', 0),
                    'closest_call': features.get('closest_call', 0),
                    'closest_call_travel_cost': features.get('closest_call_travel_cost', 0),
                    'closest_call_time_compatible': features.get('closest_call_time_compatible', 0),
                    'closest_call_size_ratio': features.get('closest_call_size_ratio', 1.0),
                    'vehicle_utilization': features.get('vehicle_utilization', 0),
                    # Add required columns with default values
                    'improvement': 0,
                    'improvement_value': 0,
                    'iteration': self.current_iteration
                }
                features_list.append(basic_features)
            except Exception as e:
                print(f"Error extracting features for call {call}: {e}")
            
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        
        try:
            # Use a simpler approach to handle predictions
            try:
                predictions = self.model.predict(df)
            except Exception as e:
                # If there are missing columns, use a simplified approach
                print(f"Prediction error: {e}")
                # Just use random selection as fallback
                return list(np.random.choice(list(calls), size=min(n_calls, len(calls)), replace=False))
            
            # Sort calls by predicted improvement
            df['predicted_improvement'] = predictions
            df_sorted = df.sort_values('predicted_improvement', ascending=False)
            
            # Apply temperature: probability-based selection with focus on top predictions
            # This increases exploration compared to always picking the top N calls
            temperature = 2.0  # Higher temperature = more randomness
            weights = np.exp(df_sorted['predicted_improvement'] / temperature)
            weights = weights / weights.sum()  # Normalize to probabilities
            
            # Turn into cumulative probabilities for weighted sampling
            cum_weights = np.cumsum(weights)
            
            # Select n_calls using weighted probabilities
            selected_calls = []
            remaining_calls = df_sorted['call_id'].values.tolist()
            
            while len(selected_calls) < n_calls and remaining_calls:
                rand_val = np.random.random()
                idx = np.searchsorted(cum_weights, rand_val)
                if idx < len(remaining_calls):
                    selected_calls.append(remaining_calls[idx])
                    # Remove selected call from remaining options
                    remaining_calls.pop(idx)
                    # Recalculate weights without this call
                    if remaining_calls:
                        weights = weights[:idx].tolist() + weights[idx+1:].tolist()
                        weights = np.array(weights) / sum(weights)  # Normalize
                        cum_weights = np.cumsum(weights)
            
            return selected_calls
        except Exception as e:
            print(f"Error making predictions: {e}")
            # Fall back to random selection
            return list(np.random.choice(list(calls), size=min(n_calls, len(calls)), replace=False))
    
    def _select_call_groups_by_ml(self, solution, data, n_total_calls):
        """Select groups of calls to remove based on group prediction model."""
        if not AUTOGLUON_AVAILABLE or self.group_model is None:
            return None
        
        # Get all unique calls in the solution
        calls = list(set(int(x) for x in solution if x > 0))
        if not calls:
            return []
            
        # Generate candidate groups
        candidate_groups = []
        
        # Add groups of various sizes (1, 2, 3)
        for size in range(1, min(self.max_group_size + 1, len(calls) + 1)):
            # Limit the number of combinations for larger call sets
            max_combinations = 100 if len(calls) > 20 else 500
            
            # Generate random combinations if there are too many possible combinations
            if len(calls) > 10 and size > 1:
                # Generate random subset of combinations
                combinations = []
                for _ in range(min(max_combinations, len(calls))):
                    group = random.sample(calls, min(size, len(calls)))
                    combinations.append(group)
            else:
                # Generate all combinations
                combinations = list(itertools.combinations(calls, size))
                # If too many combinations, sample randomly
                if len(combinations) > max_combinations:
                    combinations = random.sample(combinations, max_combinations)
            
            for combo in combinations:
                candidate_groups.append(list(combo))
        
        # Create features for each group
        group_features = []
        for group in candidate_groups:
            try:
                features = self._get_group_features(group, solution, data)
                
                # Use direct list representation for the calls instead of string
                sorted_group = sorted(group)
                
                group_features.append({
                    'calls': ' '.join(str(c) for c in sorted_group),  # Format as space-separated numbers
                    'group_size': len(group),
                    'improvement': 0,  # Dummy value
                    'improvement_value': 0,  # Dummy value
                    'iteration': self.current_iteration,
                    **{k: v for k, v in features.items() if not isinstance(v, (list, dict, np.ndarray))}
                })
            except Exception as e:
                print(f"Error creating group features: {e}")
        
        if not group_features:
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(group_features)
        
        try:
            # Make predictions
            try:
                predictions = self.group_model.predict(df)
            except Exception as e:
                print(f"Group prediction error: {e}")
                # Just use random selection as fallback
                selected_calls = []
                random_groups = random.sample(candidate_groups, min(5, len(candidate_groups)))
                for group in random_groups:
                    new_calls = [c for c in group if c not in selected_calls]
                    if len(selected_calls) + len(new_calls) <= n_total_calls:
                        selected_calls.extend(new_calls)
                
                if len(selected_calls) < n_total_calls:
                    remaining = [c for c in calls if c not in selected_calls]
                    selected_calls.extend(random.sample(remaining, min(n_total_calls - len(selected_calls), len(remaining))))
                
                return selected_calls[:n_total_calls]
            
            # Add predictions to DataFrame
            df['predicted_improvement'] = predictions
            df_sorted = df.sort_values('predicted_improvement', ascending=False)
            
            # Get top groups that don't exceed n_total_calls
            selected_calls = []
            
            # Apply temperature-based selection
            temperature = 2.0
            weights = np.exp(df_sorted['predicted_improvement'] / temperature)
            weights = weights / weights.sum()  # Normalize to probabilities
            
            # Convert to cumulative weights for sampling
            cum_weights = np.cumsum(weights)
            
            # Select groups by weighted probability until we have enough calls
            attempts = 0
            while len(selected_calls) < n_total_calls and attempts < 100:
                attempts += 1
                
                # Sample a group based on weights
                rand_val = np.random.random()
                idx = np.searchsorted(cum_weights, rand_val)
                if idx < len(df_sorted):
                    group_str = df_sorted.iloc[idx]['calls']
                    # Parse space-separated integers
                    group = [int(x) for x in group_str.split()]
                    
                    # Check if adding this group would exceed n_total_calls
                    new_calls = [c for c in group if c not in selected_calls]
                    if len(selected_calls) + len(new_calls) <= n_total_calls:
                        selected_calls.extend(new_calls)
            
            # Ensure we have the correct number of calls
            if len(selected_calls) < n_total_calls:
                # Add more individual calls if needed
                remaining_calls = [c for c in calls if c not in selected_calls]
                additional_needed = min(n_total_calls - len(selected_calls), len(remaining_calls))
                if additional_needed > 0 and remaining_calls:
                    selected_calls.extend(random.sample(remaining_calls, additional_needed))
            
            return selected_calls
        except Exception as e:
            print(f"Error making group predictions: {e}")
            # Fall back to random selection
            return list(np.random.choice(list(calls), size=min(n_total_calls, len(calls)), replace=False))
    
    def __call__(self, solution, data, seed=None, min_remove=1, max_remove=15):
        """
        Remove calls from the solution using ML-based selection or random selection.
        
        Args:
            solution: Current solution
            data: Problem data dictionary
            seed: Random seed
            min_remove: Minimum number of calls to remove
            max_remove: Maximum number of calls to remove
            
        Returns:
            Updated solution, list of removed calls
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Convert to list if numpy array
        if isinstance(solution, np.ndarray):
            solution = solution.tolist()
        
        solution_copy = deepcopy(solution)
        
        # Find all calls in the solution (excluding 0s)
        calls = set(int(x) for x in solution if x > 0)
        if not calls:
            return solution, []
        
        # Determine number of calls to remove - between 1 and 15 as specified
        n_to_remove = np.random.randint(min_remove, min(max_remove + 1, len(calls) + 1))
        
        # Decay exploration rate over time (if not in training mode)
        if not self.training_mode:
            self.exploration_rate = max(self.min_exploration_rate, 
                                        self.exploration_rate * self.exploration_decay)
        
        # Random exploration with probability exploration_rate
        if np.random.random() < self.exploration_rate:
            calls_to_move = list(np.random.choice(list(calls), size=n_to_remove, replace=False))
        else:
            # Try ML-based selection (individual calls or groups)
            calls_to_move = []
            if not self.training_mode and self.model is not None:
                # Try group selection if enabled and group model exists
                if self.enable_group_prediction and self.group_model is not None and np.random.random() < 0.5:
                    group_selected_calls = self._select_call_groups_by_ml(solution, data, n_to_remove)
                    if group_selected_calls is not None and len(group_selected_calls) > 0:
                        calls_to_move = group_selected_calls
                
                # If group selection didn't work, try individual call selection
                if not calls_to_move:
                    ml_selected_calls = self._select_calls_by_ml(solution, data, n_to_remove)
                    if ml_selected_calls is not None and len(ml_selected_calls) > 0:
                        calls_to_move = ml_selected_calls
            
            # Fall back to random selection if ML selection failed or in training mode
            if not calls_to_move:
                calls_to_move = list(np.random.choice(list(calls), size=n_to_remove, replace=False))
        
        # Remove the selected calls from the solution
        solution = solution_copy  # Reset to original solution
        for call in calls_to_move:
            # Find positions of the call
            if call not in solution:
                continue
                
            pickup_idx = solution.index(call)
            delivery_idx = len(solution) - 1 - solution[::-1].index(call)
            
            # Remove the call from both positions
            solution.pop(delivery_idx)
            solution.pop(pickup_idx)
        
        return solution, calls_to_move


# Create a global instance that persists between calls
# Initialize it here rather than in the function to avoid the NoneType error
_ml_removal_operator = MLRemovalOperator(training_mode=True, training_iterations=1200, retrain_at=9000)


def ml_removal(solution, data, seed=None, min_remove=1, max_remove=15):
    """
    ML-based removal operator with fallback to random removal.
    
    This is a wrapper function that uses a global MLRemovalOperator instance.
    
    Args:
        solution: Current solution
        data: Problem data dictionary
        seed: Random seed
        min_remove: Minimum number of calls to remove (default: 1)
        max_remove: Maximum number of calls to remove (default: 15)
        
    Returns:
        Updated solution, list of removed calls
    """
    # Use module-level variable to maintain state between calls
    global _ml_removal_operator
    
    # Apply the operator (it's already initialized above)
    return _ml_removal_operator(solution, data, seed, min_remove, max_remove)


def hybrid_removal(solution, data, seed=None, min_remove=1, max_remove=15):
    """
    Hybrid removal operator that combines ML-based and random removal.
    
    This operator randomly chooses between ML-based removal and pure random removal,
    allowing for both exploitation of learned patterns and exploration through randomness.
    
    Args:
        solution: Current solution
        data: Problem data dictionary
        seed: Random seed
        min_remove: Minimum number of calls to remove
        max_remove: Maximum number of calls to remove
        
    Returns:
        Updated solution, list of removed calls
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 70% chance to use ML removal, 30% chance for completely random removal
    # This ratio can be adjusted based on performance
    use_ml = np.random.random() < 0.7
    
    if use_ml:
        # Use ML removal
        global _ml_removal_operator
        return _ml_removal_operator(solution, data, seed, min_remove, max_remove)
    else:
        # Import random_removal at runtime to avoid circular imports
        from src.operators.random_removal import random_removal
        # Use pure random removal
        return random_removal(solution, data, seed, min_remove, max_remove)