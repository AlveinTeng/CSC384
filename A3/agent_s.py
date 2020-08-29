"""
An AI player for Othello.
"""

import random
import sys
import time

# You can use the functions in othello_shared to write your AI
from othello_shared import find_lines, get_possible_moves, get_score, play_move


# Caching dictionary
caching_dir = {}

def eprint(*args, **kwargs): #you can use this for debugging, as it will print to sterr and not stdout
    print(*args, file=sys.stderr, **kwargs)

# Method to compute utility value of terminal state
def compute_utility(board, color):
    #IMPLEMENT

    if color == 1:
        return get_score(board)[0] - get_score(board)[1]
    else:
        return get_score(board)[1] - get_score(board)[0]
# Better heuristic value of board
# def compute_heuristic(board, color): #not implemented, optional
#     #IMPLEMENT
#
#     utility = compute_utility(board, color)
#
#     # Consider board locations where pieces are stable (e.g. corner)
#     stable_loc = compute_stable(board, color)
#
#     # Consider the number of moves you and your opponent can make given the current board configuration.
#     moves = compute_move(board, color)
#
#     # Use a different strategy in the opening, mid-game, and end-game.
#     heur = utility + stable_loc + moves
#
#     return heur

# def compute_stable(board, color):
#     corners = [board[0][-1], board[0][0], board[-1][-1], board[-1][0]]
#
#     score_self, score_oppo = 0, 0
#
#     for i in range(corners):
#         if corners[i] != 0:
#             horizontal = compute_horizontal(board, corners[i], i)
#             vertical = compute_vertical(board, corners[i], i)
#             if corners[i] == color:
#                 if not (horizontal == len(board) - 2 and ((i == 1 and corners[0] == corners[i]) or (i == 3 and corners[2] == corner[i]))):
#                     score_self += horizontal
#                 if not (vertical == len(board) - 2 and ((i == 2 and corners[0] == corners[i]) or (i == 3 and corners[1] == corners[i]))):
#                     score_self += vertical
#                 score_self += 1
#             else:
#                 if not (horizontal == len(board) - 2 and ((i == 1 and corners[0] == corners[i]) or (i == 3 and corners[2] == corner[i]))):
#                     score_oppo += horizontal
#                 if not (vertical == len(board) - 2 and ((i == 2 and corners[0] == corners[i]) or (i == 3 and corners[1] == corners[i]))):
#                     score_oppo += vertical
#                 score_oppo += 1
#
#     if color == 1:  # dark disk
#         return score_self - score_oppo
#     return score_oppo - score_self  # light disk

# corner: 0 -> up_right, 1 -> up_left, 2 -> down_right, 3 -> down_left
# def compute_horizontal(board, color, corner):
#     count = 0
#     if corner == 0:
#         col = len(board) - 2
#
#         while col > 0:
#             if board[0][col] == color:
#                 count += 1
#             else:
#                 break
#             col -= 1
#
#     elif corner == 1:
#         col = 1
#
#         while col < len(board) - 1:
#             if board[0][col] == color:
#                 count += 1
#             else:
#                 break
#             col += 1
#
#     elif corner == 2:
#         col = len(board) - 2
#
#         while col > 0:
#             if board[-1][col] == color:
#                 count += 1
#             else:
#                 break
#             col -= 1
#
#     elif corner == 3:
#         col = 1
#
#         while col < len(board) - 1:
#             if board[-1][col] == color:
#                 count += 1
#             else:
#                 break
#             col += 1
#
#     return count

# corner: 0 -> up_right, 1 -> up_left, 2 -> down_right, 3 -> down_left
# def compute_vertical(board, color, corner):
#     count = 0
#     if corner == 0:
#         row = 1
#
#         while row < len(board) - 1:
#             if board[row][-1] == color:
#                 count += 1
#             else:
#                 break
#             row += 1
#     elif corner == 1:
#         row = 1
#
#         while row < len(board) - 1:
#             if board[row][0] == color:
#                 count += 1
#             else:
#                 break
#             row += 1
#     elif corner == 2:
#         row = len(board) - 2
#
#         while row > 0:
#             if board[row][-1] == color:
#                 count += 1
#             else:
#                 break
#             row -= 1
#     elif corner == 3:
#         row = len(board) - 2
#
#         while row > 0:
#             if board[row][-1] == color:
#                 count += 1
#             else:
#                 break
#             row -= 1
#
#     return count
#
# def compute_move(board, color):
#     dark_moves, light_moves = len(get_possible_moves(board, 1)), len(get_possible_moves(board, 2))
#
#     if color == 1:  # dark disk
#         return dark_moves - light_moves
#     return light_moves - dark_moves # light disk

############ MINIMAX ###############################
def minimax_min_node(board, color, limit, caching = 0):
    #IMPLEMENT

    global caching_dir

    state = board, color

    if caching == 1 and state in caching_dir:
        return caching_dir[state]

    if limit == 0:
        return (0, 0), compute_utility(board, color)

    possible_moves = get_possible_moves(board, 3 - color)

    if len(possible_moves) == 0:
        new_state = (0, 0), compute_utility(board, color)

        if caching == 1:
            caching_dir[state] = new_state

        return new_state

    min_move = (0, 0)
    min_util = float("inf")

    for move in possible_moves:
        new_board = play_move(board, 3 - color, move[0], move[1])
        temp_move, temp_util = minimax_max_node(new_board, color, limit - 1, caching)
        if temp_util < min_util:
            min_util = temp_util
            min_move = move

    new_state = min_move, min_util

    if caching == 1:
        caching_dir[state] = new_state

    return new_state


def minimax_max_node(board, color, limit, caching = 0): #returns highest possible utility
    #IMPLEMENT

    global caching_dir

    state = board, color

    if caching == 1 and state in caching_dir:
        return caching_dir[state]

    if limit == 0:
        return (0, 0), compute_utility(board, color)

    possible_moves = get_possible_moves(board, color)

    if len(possible_moves) == 0:
        new_state = (0, 0), compute_utility(board, color)

        if caching == 1:
            caching_dir[state] = new_state

        return new_state

    max_move = (0, 0)
    max_util = -float("inf")

    for move in possible_moves:
        new_board = play_move(board, color, move[0], move[1])
        temp_move, temp_util = minimax_min_node(new_board, color, limit - 1, caching)
        if temp_util > max_util:
            max_util = temp_util
            max_move = move

    new_state = max_move, max_util

    if caching == 1:
        caching_dir[state] = new_state

    return new_state


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

    move, utility = minimax_max_node(board, color, limit, caching)

    return move

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
        possible_moves = node_ordering_heur(get_possible_moves(board, 3 - color), board, 3 - color, color, False)
    else:
        possible_moves = get_possible_moves(board, 3 - color)

    if len(possible_moves) == 0:
        new_state = (0, 0), compute_utility(board, color)

        if caching == 1:
            caching_dir[state] = new_state

        return new_state

    min_move = (0, 0)
    min_util = float("inf")

    for move in possible_moves:
        new_board = play_move(board, 3 - color, move[0], move[1])
        temp_move, temp_util = alphabeta_max_node(new_board, color, alpha, beta, limit - 1, caching, ordering)
        if temp_util < min_util:
            min_util = temp_util
            min_move = move
        if beta > min_util:
            beta = min_util
            if beta <= alpha:
                break

    new_state =  min_move, min_util

    if caching == 1:
        caching_dir[state] = new_state

    return new_state

def alphabeta_max_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    #IMPLEMENT

    global caching_dir

    state = board, color

    if caching == 1 and state in caching_dir:
        return caching_dir[state]

    if limit == 0:
        return (0, 0), compute_utility(board, color)

    if ordering == 1:
        possible_moves = node_ordering_heur(get_possible_moves(board, color), board, color, color, True)
    else:
        possible_moves = get_possible_moves(board, color)

    if len(possible_moves) == 0:
        new_state = (0, 0), compute_utility(board, color)

        if caching == 1:
            caching_dir[state] = new_state

        return new_state

    max_move = (0, 0)
    max_util = -float("inf")

    for move in possible_moves:
        new_board = play_move(board, color, move[0], move[1])
        temp_move, temp_util = alphabeta_min_node(new_board, color, alpha, beta, limit - 1, caching, ordering)
        if temp_util > max_util:
            max_util = temp_util
            max_move = move
        if alpha < max_util:
            alpha = max_util
            if beta <= alpha:
                break

    new_state = max_move, max_util

    if caching == 1:
        caching_dir[state] = new_state

    return new_state

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

    move, utility = alphabeta_max_node(board, color, -float('inf'), float('inf'), limit, caching, ordering)

    return move

def node_ordering_heur(possible_moves, board, d_color, color, r):
    return sorted(possible_moves, key= lambda move: compute_utility(play_move(board, d_color, move[0], move[1]), color), reverse = r)

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
