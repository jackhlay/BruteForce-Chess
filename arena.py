import asyncio
import json
import requests
import random
import time
import websockets

#Lichess api
import berserk

import bulletchess as bc
import bulletchess.utils as bcu

#Engin
from engin import find_best_move, tt_size, tt_clear, static_eval


# UTIL #
#########################################################################################################

def seekGames(api_tok):
    games = 0
    while games < 3:
        session = berserk.TokenSession(api_tok)
        board = berserk.clients.Board(session=session, base_url="https://lichess.org")
        board.stream_incoming_events()
        games += 1

def makeGame(fen = None):
    if fen:
        board = bc.Board.from_fen(fen)
    else:
        board = bc.Board()
        fen = board.fen
    
    moves = board.legal_moves()
    color = board.turn
    opp = color.opposite

def sendpos(board: bc.Board, res):
    static = static_eval(board)
    uri = "http://localhost:5000/api/update"
    payload = {
        "fen": board.fen(),
        "evalSource1": f"{bcu.evaluate(board)}",
        "evalSource2": f"{(-1 * static) if board.turn==bc.BLACK else static}",
        "results": res
    }
    try:
        response = requests.post(uri, json=payload)  # Use json=payload
        # [print](response.status_code, response.text)
        # print(f"pos sent! {response}, fen: {board.fen()}")
    except Exception:
        print("Unable to send to front end")

async def query_stockfish(fen: str, depth=3):
    uri = "ws://localhost:4000/"
    async with websockets.connect(uri) as ws:
        # Send position and go
        await ws.send(json.dumps({"type": "uci:command", "payload": f"position fen {fen}"}))
        await ws.send(json.dumps({"type": "uci:command", "payload": f"go depth {depth}"}))
        # Wait for bestmove response
        async for msg in ws:
            # print(msg)
            data = json.loads(msg)
            if data.get("type") == "uci:response" and ("bestmove" in data.get("payload")):
                return data["payload"].split(" ")[1]

def get_sfmove(fen: str, depth=3):
    res = asyncio.run(query_stockfish(fen, depth))
    print(f"RES: {res}")
    return res

def isTerminal(board:bc.Board):

    terminal_statuses = [bc.MATE, bc.CHECKMATE, bc.DRAW]

    for s in terminal_statuses:
        if board in s:
            return True
    return False

############################################################################

searchTimeLimit = 2
res = {"White" : 0 , 
    "Draw": 0,
    "Black": 0,
    "???" : 0}

for i in range(128):
    board = bc.Board()
    for _ in range(6):
        board.apply(random.choice(list(board.legal_moves())))
        time.sleep(.34)

    while not isTerminal(board):
        if board.turn == bc.WHITE:
            bestMove = find_best_move(board, searchTimeLimit, 999)
            # bestMove = bc.Move.from_uci(get_sfmove(board.fen(), 1))
            board.apply(bestMove)
            sendpos(board, res)

        elif board.turn == bc.BLACK:
            # bestMove = find_best_move(board, searchTimeLimit, 999)
            bestMove = bc.Move.from_uci(get_sfmove(board.fen(),1))
            board.apply(bestMove)
            sendpos(board, res)

   
    if (board in bc.DRAW) or (board in bc.FORCED_DRAW):
        res["Draw"] += 1
    elif (board.turn == bc.WHITE) and (board in bc.CHECKMATE):
        res["Black"] += 1
    elif (board.turn == bc.BLACK) and (board in bc.CHECKMATE):
        res["White"] += 1
    else:
        res["???"] += 1

    size, lngth = tt_size()

    print(f"RESULTS SO FAR: WHITE (engin - {searchTimeLimit} seconds search): {res['White']}, DRAW: {res['Draw']}, BLACK (SF DEPTH 1): {res['Black']} ???: {res['???']}")
    print(f"TTABLE SIZE : {size} MB, {lngth} ENTRIES")
    if size > 1024: 
        tt_clear()
        print ("CLEARED TT")