import chess
from tree import get_node, mcts
import random
import requests
import time

def sendpos(fen: str):
    uri = "http://localhost:5000/api/update"
    payload = {
        "fen":fen,
        "evalSource1": "N/A",
        "evalSource2": "N/A" 

    }

    response= requests.post(uri, data=payload)

def play_game(iterations=40):
    board = chess.Board()
    root = get_node(board)

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            best_move = mcts(root, iterations=iterations)
            board.push(best_move)
            fen = board.fen()

        else: 
            move = random.choice(list(board.legal_moves))
            board.push(move)
            fen = board.fen()
    sendpos(fen)
    time.sleep(5)


    print("Game over:", board.result())



if __name__ == "__main__":
    play_game()