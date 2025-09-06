import chess
from tree import get_node, mcts
import random
import requests
import time

def sendpos(fen: str):
    uri = "http://127.0.0.1:5000/api/update"
    payload = {
        "fen": fen,
        "evalSource1": "N/A",
        "evalSource2": "N/A"
    }
    response = requests.post(uri, json=payload)  # Use json=payload
    print(response.status_code, response.text)
    print(f"pos sent! {response}")

def play_game(iterations=400):
    board = chess.Board()
    root = get_node(board)
    current = get_node(board)
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            best_move = mcts(current, iterations=1300)
            board.push(best_move)
            current = get_node(board)
            fen = board.fen()
            print(board.fen())
            [print(f"\n {board} \n")]
            sendpos(fen)        



        else: 
            time.sleep(.7)
            move = random.choice(list(board.legal_moves))
            board.push(move)
            current = get_node(board)
            fen = board.fen()
            print(fen)
            [print(f"\n{board}")]
            sendpos(fen)

    sendpos(fen)        


    print("Game over:", board.result())



if __name__ == "__main__":
    play_game()