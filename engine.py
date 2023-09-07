from board import Board
import random

def find_best_move(board):
    print(board.get_legal_moves())
    moves = list(board.get_legal_moves())
    idx = random.randint(0, len(moves) - 1)
    return moves[idx]
