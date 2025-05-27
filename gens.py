import asyncio
import chess
import json
import os
import random
import re
from redis import Redis
from time import sleep
import websockets

uri = "ws://localhost:4000"

async def askStockFish(fen:str) -> float:
    posCom = { "type": "uci:command", "payload": "position fen " + fen }
    sleep(.5)
    goCom = { "type": "uci:command", "payload": "go depth 3" }
    sleep(.5)
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps(posCom))
        await ws.send(json.dumps(goCom))
        while True:
            res = await ws.recv()
            res_data = json.loads(res)
            if res_data.get("type") == "uci:response" and (("info depth 3" and "cp") in res_data['payload']):
                match = re.search(r"score cp (-?\d+)", res_data['payload'])
                score = int(match.group(1))/100
                print(f"Stockfish response: {res_data['payload']}")
                print(f"Score: {score}")

                return 0.0
            else:
                continue



async def generate(red: Redis):
    board = chess.Board()
    moves = list(board.legal_moves)
    depth = 7.0

    for move in moves:
        board.push_san(str(move))
        postFen = board.fen()
        rating = await askStockFish(postFen)
        payload = {"fen":postFen, "rating":rating}
        red.sadd("BRUTE", json.dumps(payload))
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
    depth = 7500.0

    while depth >= 0.000001:
        moves = list(board.legal_moves)
        move = moves[random.randint(0, len(moves)-1)]
        board.push_san(str(move))
        postFen = board.fen()
        rating = await askStockFish(postFen)
        payload = {"fen":postFen}
        # payload = {"fen":postFen, "rating":64}
        val = red.sadd("RANDO", json.dumps(payload))
        if board.is_game_over() or len(moves)==0:
            board=chess.Board()
        depth-=.5

red = Redis(
    host = os.getenv("REDHOST","localhost"),
    port = os.getenv("REDPORT",6379)
    )

async def main():
    await rando(red)

asyncio.run(main())