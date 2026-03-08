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

def sendpos(board: bc.Board):
    uri = "http://localhost:5000/api/update"
    payload = {
        "fen": board.fen(),
        "evalSource1": f"{bcu.evaluate(board)}",
        "evalSource2": "N/A"
    }
    response = requests.post(uri, json=payload)  # Use json=payload
    # print(response.status_code, response.text)
    # print(f"pos sent! {response}, fen: {board.fen()}")

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

#########################################################################################################

node_count = 0

def isTerminal(board:bc.Board):

    terminal_statuses = [bc.MATE, bc.CHECKMATE, bc.DRAW]

    for s in terminal_statuses:
        if board in s:
            return True
    return False

def scoreMoves(move: bc.Move, board: bc.Board):
    board.apply(move)
    score = bcu.evaluate(board)
    board.undo()
    return score

def quiesce(board: bc.Board, alpha, beta):
    global node_count

    if bcu.is_quiescent(board):
        return bcu.evaluate(board)

    standPat = bcu.evaluate(board)
    if standPat >= beta:
        return beta
    if standPat > alpha:
        alpha = standPat

    for move in board.legal_moves():
        if move.is_capture(board):
            board.apply(move)
            score = -quiesce(board, -alpha, -beta)
            node_count += 1
            board.undo()
    
        if score >= beta:
            return score
        if score > alpha:
            alpha = score
    return alpha

# ADVANCED EVAL
def negaMax(board: bc.Board, depth, alpha, beta):
    global node_count
    if (depth == 0) or (isTerminal(board)):
        return bcu.evaluate(board)
    
    moves = list(board.legal_moves())
    moves.sort(key=lambda m: scoreMoves(m, board), reverse=True)
    value = float("-inf")

    for m in moves:
        board.apply(m)
        node_count += 1
        score = -negaMax(board, depth-1, -beta, -alpha)
        board.undo()
        if score > value:
            value = score
        alpha = max(alpha, value)
        if alpha > beta:
            break

    return value

def rootEval(board: bc.Board, depth):
    global node_count
    alpha = float("-inf")
    beta = float("inf")
    best_value = alpha
    bestMove = None
    node_count = 0

    for m in board.legal_moves():
        board.apply(m)
        score = -negaMax(board, depth-1,-beta,-alpha)
        node_count += 1
        board.undo()
        if score > best_value:
            best_value = score
            bestMove = m
        alpha = max(alpha,best_value)
        if alpha > beta:
            break

    return bestMove




# ~~~~~~~~~~~~~~~~ #


res = {"White" : 0 , 
    "Draw": 0,
    "Black": 0,
    "???" : 0}

for i in range(37):
    board = bc.Board()
    board.apply(random.choice(list(board.legal_moves())))
    sendpos(board)
    board.apply(random.choice(list(board.legal_moves())))
    sendpos(board)

    while not isTerminal(board):
        if board.turn == bc.WHITE:
            start = time.time()
            bestMove = rootEval(board, 5)
            dur = time.time() - start
            board.apply(bestMove)
            sendpos(board)

        if board.turn == bc.BLACK:
            start = time.time()
            bestMove = rootEval(board, 4)
            dur = time.time() - start
            board.apply(bestMove)
            sendpos(board)
            pass

   
    if board in bc.DRAW:
        res["Draw"] += 1
    elif (board.turn == bc.WHITE) and (board in bc.CHECKMATE):
        res["Black"] += 1
    elif (board.turn == bc.BLACK) and (board in bc.CHECKMATE):
        res["White"] += 1
    else:
        res["???"] += 1

    print(f"RESULTS SO FAR: WHITE (DEPTH 4): {res['White']}, DRAW: {res['Draw']}, BLACK (DEPTH 4): {res['Black']} ???: {res['???']}")

