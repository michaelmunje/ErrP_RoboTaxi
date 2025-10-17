import numpy as np


# class CellType(object):
#     """ Defines all types of cells that can be found in the game. """

#     EMPTY = 0
#     GOOD_FRUIT = 1
#     BAD_FRUIT = 2
#     LAVA = 3
#     SNAKE_HEAD = 4
#     SNAKE_BODY = 5
#     WALL = 6
#     PIT = 7
#     COLLABORATOR_HEAD = 8
#     COLLABORATOR_BODY = 9

# prev_state and current_state are 8x8 numpy arrays

def manhattan_distance(pos2d_1, pos2d_2):
    return abs(pos2d_1[0] - pos2d_2[0]) + abs(pos2d_1[1] - pos2d_2[1])

def inverse_manhattan_distance(pos2d_1, pos2d_2, epsilon = 1):
    return 1 / (abs(pos2d_1[0] - pos2d_2[0]) + abs(pos2d_1[1] - pos2d_2[1]) + epsilon)

def get_pos_tuple_where(state, value):
    assert state.shape == (8, 8), "state must be a 8x8 numpy array"
    tmp =  np.where(state == value) # (array([2]), array([2])) or (array([2,3]), array([2,3]))
    return list(zip(tmp[0].tolist(), tmp[1].tolist())) # [(1, 2), (2, 2)]

def get_snake_head_position(state): # return a list of tuples of length 1
    return get_pos_tuple_where(state, 4)

def get_passenger_position(state): # return a list of tuples of length 0-2 inclusive
    return get_pos_tuple_where(state, 1)

def get_obstacle_position(state): # return a list of tuples of length 0-2 inclusive
    return get_pos_tuple_where(state, 3)

def compute_delta_features_v3(prev_state, current_state):
    previous_passenger_positions = get_passenger_position(prev_state)
    previous_obstacle_positions = get_obstacle_position(prev_state)
    assert len(previous_passenger_positions) <= 2, "at most two passengers allowed"
    assert len(previous_obstacle_positions) <= 2, "at most two obstacles allowed"
    
    current_snake_head_position = get_snake_head_position(current_state)
    assert len(current_snake_head_position) == 1, "Only One Head Allowed"
    previous_snake_head_position = get_snake_head_position(prev_state)
    assert len(previous_snake_head_position) == 1, "Only One Head Allowed"
    
    current_snake_head_position = current_snake_head_position[0]
    previous_snake_head_position = previous_snake_head_position[0]

    collision_passenger = int(current_snake_head_position in previous_passenger_positions)
    collision_obstacle = int(current_snake_head_position in previous_obstacle_positions)
    
    passenger_proximity_score_prev = sum(inverse_manhattan_distance(previous_snake_head_position, p) for p in previous_passenger_positions)
    obstacle_proximity_score_prev = sum(inverse_manhattan_distance(previous_snake_head_position, o) for o in previous_obstacle_positions)
    
    passenger_proximity_score_curr = sum(inverse_manhattan_distance(current_snake_head_position, p) for p in previous_passenger_positions)
    obstacle_proximity_score_curr = sum(inverse_manhattan_distance(current_snake_head_position, o) for o in previous_obstacle_positions)
    
    passenger_proximity_score_delta = passenger_proximity_score_curr - passenger_proximity_score_prev
    obstacle_proximity_score_delta = obstacle_proximity_score_curr - obstacle_proximity_score_prev
    
    return np.array([collision_passenger, collision_obstacle, passenger_proximity_score_delta, obstacle_proximity_score_delta, 0, 0])

def compute_delta_features_v6(prev_state, current_state):
    """
    returns 6 dimensions, where all dimensions are 0 except for 2,3, 
    which are delta min(manhattan_distance to positive target) 
    and delta min(manhattan_distance to negative target)
    """
    previous_passenger_positions = get_passenger_position(prev_state)
    previous_obstacle_positions = get_obstacle_position(prev_state)
    assert len(previous_passenger_positions) <= 2, "at most two passengers allowed"
    assert len(previous_obstacle_positions) <= 2, "at most two obstacles allowed"
    
    current_snake_head_position = get_snake_head_position(current_state)
    assert len(current_snake_head_position) == 1, "Only One Head Allowed"
    previous_snake_head_position = get_snake_head_position(prev_state)
    assert len(previous_snake_head_position) == 1, "Only One Head Allowed"
    
    current_snake_head_position = current_snake_head_position[0]
    previous_snake_head_position = previous_snake_head_position[0]

    collision_passenger = int(current_snake_head_position in previous_passenger_positions)
    collision_obstacle = int(current_snake_head_position in previous_obstacle_positions)
    
    passenger_proximity_score_prev = min(manhattan_distance(previous_snake_head_position, p) for p in previous_passenger_positions) if len(previous_passenger_positions) > 0 else 0
    obstacle_proximity_score_prev = min(manhattan_distance(previous_snake_head_position, o) for o in previous_obstacle_positions) if len(previous_obstacle_positions) > 0 else 0
    
    passenger_proximity_score_curr = min(manhattan_distance(current_snake_head_position, p) for p in previous_passenger_positions) if len(previous_passenger_positions) > 0 else 0
    obstacle_proximity_score_curr = min(manhattan_distance(current_snake_head_position, o) for o in previous_obstacle_positions) if len(previous_obstacle_positions) > 0 else 0
    
    scale = 0.8
    
    passenger_proximity_score_delta = passenger_proximity_score_curr**scale - passenger_proximity_score_prev**scale
    obstacle_proximity_score_delta = obstacle_proximity_score_curr**scale - obstacle_proximity_score_prev**scale
    
    passenger_proximity_score_delta = passenger_proximity_score_delta *2
    obstacle_proximity_score_delta = obstacle_proximity_score_delta *2
    
    return np.array([0, 0, passenger_proximity_score_delta, obstacle_proximity_score_delta, 0, 0])

def compute_delta_features_v4(prev_state, current_state):
    """
    returns 6 dimensions, where all dimensions are 0 except for 2,3, 
    which are delta sum(1/manhattan_distance to positive target) 
    and delta sum(1/manhattan_distance to negative target)
    """
    delta_fn = compute_delta_features_v3(prev_state, current_state)
    delta_fn[0] = 0
    delta_fn[1] = 0
    return delta_fn

def compute_delta_features_v2(prev_state, current_state):
    old_fn = preprocess_observation_tamer(observation)
    new_fn = preprocess_observation_tamer(simulated_observation)
    new_fn, old_fn = handle_collision(new_fn, old_fn)
    delta_fn = new_fn - old_fn
    return np.array(delta_fn)


def handle_collision(f_curr, f_prev):
    if f_curr[0] < f_prev[0]: # crash happened
        f_curr[2] = 0
        f_curr[4] = 0
    if f_curr[1] < f_prev[1]: 
        f_curr[3] = 0
        f_curr[5] = 0
    return f_curr, f_prev

