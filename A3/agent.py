"""
An AI player for Othello.
"""

import random
import sys
import time

# You can use the functions in othello_shared to write your AI
from othello_shared import find_lines, get_possible_moves, get_score, play_move

#caching dir
caching_dir = {}


def eprint(*args, **kwargs): #you can use this for debugging, as it will print to sterr and not stdout
    print(*args, file=sys.stderr, **kwargs)

# Method to compute utility value of terminal state


def compute_utility(board, color):
    if color == 1:
        return get_score(board)[0] - get_score(board)[1]
    else:
        return get_score(board)[1] - get_score(board)[0]
    # return  utility #change this!

# Better heuristic value of board


def compute_heuristic(board, color): #not implemented, optional
    #IMPLEMENT
    '''

    I am going to calculate the heuristic as
    the number of moves you and your opponent can make given the current board conﬁguration
    '''

    move = get_possible_moves(board, color)
    opponent_move = get_possible_moves(board, 3 - color)
    heuristic = len(move) - len(opponent_move)
    
    return heuristic #change this!


############ MINIMAX ###############################


def minimax_min_node(board, color, limit, caching = 0):

    global caching_dir

    state = board, color

    if caching == 1 and state in caching_dir:
        return caching_dir[state]

    if limit == 0:
        return (0, 0), compute_utility(board, color)

    moves = get_possible_moves(board, 3 - color)

    if len(moves) == 0:
        new_state = (0, 0), compute_utility(board, color)

        if caching == 1:
            caching_dir[state] = new_state

        return new_state

    min_move = (0,0)
    utility = float('inf')
    for move in moves:
        temp_board = play_move(board, 3 - color, move[0], move[1])
        temp_value = minimax_max_node(temp_board, color, limit - 1, caching)
        if temp_value[1] < utility:
            utility = temp_value[1]
            min_move = move
    return min_move, utility


def minimax_max_node(board, color, limit, caching = 0): #returns highest possible utility
    #IMPLEMENT

    global caching_dir

    state = board, color

    if caching == 1 and state in caching_dir:
        return caching_dir[state]

    if limit == 0:
        return (0, 0), compute_utility(board, color)

    moves = get_possible_moves(board, color)

    if len(moves) == 0:
        new_state = (0, 0), compute_utility(board, color)

        if caching == 1:
            caching_dir[state] = new_state

        return new_state

    max_move = (0, 0)
    utility = -float('inf')

    for move in moves:
        temp_board = play_move(board, color, move[0], move[1])
        temp_value = minimax_min_node(temp_board, color, limit - 1, caching)
        if temp_value[1] > utility:
            utility = temp_value[1]
            max_move = move

    return max_move, utility


def select_move_minimax(board, color, limit, caching = 0):
    """
    Given a board and a player color, decide on a move.
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.
    """
    #IMPLEMENT
    best_move = minimax_max_node(board, color, limit, caching)
    return best_move[0] #change this!

############ ALPHA-BETA PRUNING #####################


def alphabeta_min_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    #IMPLEMENT

    global caching_dir

    state = board, color

    if caching == 1 and state in caching_dir:
        return caching_dir[state]

    if limit == 0:
        return (0, 0), compute_utility(board, color)

    if ordering == 1:
        moves = node_ordering_heur_min(board, color)
    else:
        moves = get_possible_moves(board, 3 - color)

    if len(moves) == 0:
        new_state = (0, 0), compute_utility(board, color)

        if caching == 1:
            caching_dir[state] = new_state

        return new_state

    min_move = (0, 0)
    min_val = float('inf')

    for move in moves:
        temp_board = play_move(board, 3 - color, move[0], move[1])
        temp_value = alphabeta_max_node(temp_board, color, alpha, beta, limit - 1,
                                        caching, ordering)
        if temp_value[1] < min_val:
            min_val = temp_value[1]
            min_move = move
        if beta > min_val:
            beta = min_val
            if beta <= alpha:
                break

    return min_move, min_val  # change this!


def alphabeta_max_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    #IMPLEMENT
    # state = (board, color)
    # if caching == 1 and state in caching_dir:
    #     return caching_dir[state]

    global caching_dir

    state = board, color

    if caching == 1 and state in caching_dir:
        return caching_dir[state]

    if limit == 0:
        return (0, 0), compute_utility(board, color)

    if ordering == 1:
        moves = node_ordering_heur_max(board,color)
    else:
        moves = get_possible_moves(board, color)

    if len(moves) == 0:
        new_state = (0, 0), compute_utility(board, color)

        if caching == 1:
            caching_dir[state] = new_state

        return new_state

    max_move = (0, 0)
    max_val = -float('inf')
    for move in moves:
        temp_board = play_move(board, color, move[0], move[1])
        temp_value = alphabeta_min_node(temp_board, color, alpha, beta, limit - 1,
                                        caching, ordering)
        if temp_value[1] > max_val:
            max_val = temp_value[1]
            max_move = move
        if alpha < max_val:
            alpha = max_val
            if beta <= alpha:
                break
    return max_move, max_val #change this!


def select_move_alphabeta(board, color, limit, caching = 0, ordering = 0):
    """
    Given a board and a player color, decide on a move.
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.
    If ordering is ON (i.e. 1), use node ordering to expedite pruning and reduce the number of state evaluations.
    If ordering is OFF (i.e. 0), do NOT use node ordering to expedite pruning and reduce the number of state evaluations.
    """
    #IMPLEMENT
    best_move = alphabeta_max_node(board, color, -float('inf'), float('inf'), limit, caching, ordering)
    return best_move[0] #change this!


def node_ordering_heur_max(board, color):
    moves = get_possible_moves(board, color)
    sorted_moves = sorted(moves, key=lambda move: compute_utility(play_move(board, color, move[0], move[1]), color))
    sorted_moves.reverse()
    return sorted_moves


def node_ordering_heur_min(board, color):

    moves = get_possible_moves(board, 3-color)
    sorted_moves = sorted(moves, key=lambda move: compute_utility(play_move(board,  3 - color, move[0], move[1]), color))
    return sorted_moves


####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Othello AI") # First line is the name of this AI
    arguments = input().split(",")

    color = int(arguments[0]) #Player color: 1 for dark (goes first), 2 for light.
    limit = int(arguments[1]) #Depth limit
    minimax = int(arguments[2]) #Minimax or alpha beta
    caching = int(arguments[3]) #Caching
    ordering = int(arguments[4]) #Node-ordering (for alpha-beta only)

    if (minimax == 1): eprint("Running MINIMAX")
    else: eprint("Running ALPHA-BETA")

    if (caching == 1): eprint("State Caching is ON")
    else: eprint("State Caching is OFF")

    if (ordering == 1): eprint("Node Ordering is ON")
    else: eprint("Node Ordering is OFF")

    if (limit == -1): eprint("Depth Limit is OFF")
    else: eprint("Depth Limit is ", limit)

    if (minimax == 1 and ordering == 1): eprint("Node Ordering should have no impact on Minimax")

    while True: # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL": # Game is over.
            print
        else:
            board = eval(input()) # Read in the input and turn it into a Python
                                  # object. The format is a list of rows. The
                                  # squares in each row are represented by
                                  # 0 : empty square
                                  # 1 : dark disk (player 1)
                                  # 2 : light disk (player 2)

            # Select the move and send it to the manager
            if (minimax == 1): #run this if the minimax flag is given
                movei, movej = select_move_minimax(board, color, limit, caching)
            else: #else run alphabeta
                movei, movej = select_move_alphabeta(board, color, limit, caching, ordering)

            print("{} {}".format(movei, movej))

if __name__ == "__main__":
    run_ai()
