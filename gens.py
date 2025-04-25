import json
import os

import chess
import random
import websockets
from redis import Redis

uri = os.getenv("STOCKFISHURI")

async def askStockFish(fen:str) -> float:
    posCom = { "type": "uci:command", "payload": "position fen " + fen }
    goCom = { "type": "uci:command", "payload": "go depth 3" }
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps(posCom))
        await ws.send(json.dumps(goCom))
        while True:
            res = await ws.recv()
            res_data = json.loads(res)
            if res_data.get("type") == "uci:response":
                print(f"Stockfish response: {res_data['payload']}")
                return 0.0
            else: print(f"RESPONSE FROM SOMETHING ELSE - {res_data}")
            return -500.0



async def generate(red: Redis):
    board = chess.Board()
    moves = list(board.legal_moves)
    depth = 7.0

    for move in moves:
        board.push_san(str(move))
        postFen = board.fen()
        rating = askStockFish(postFen)
        payload = {"fen":postFen, "rating":rating}
        await red.sadd("BRUTE", json.dumps(payload))
        await recurse(postFen, depth-.5)
    #TODO: Implement proper recursion

async def recurse(fen, depth):
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    if depth <= 0.000001:
        print("MAX DEPTH REACHED. SHUTTING DOWN...")
        return

    for move in moves:
        board.push_san(str(move))
        nFen = board.fen()
        await recurse(nFen, depth-.5)

async def rando(red: Redis):
    board = chess.Board()
    moves = list(board.legal_moves)
    depth = 7500.0

    while depth >= 0.000001:
        move = moves[random.randint(0, len(moves))]
        board.push_san(str(move))
        postFen = board.fen()
        rating = askStockFish(postFen)
        payload = {"fen":postFen, "rating":rating}
        await red.sadd("RANDO", json.dumps(payload))
        if board.is_game_over():
            board=chess.Board()
        depth-=.5