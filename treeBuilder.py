import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import asyncio
import requests
import json
import websockets

import chess
import math
import random
import pickle
import sys
import time



iters = 537
TranspositionTable = {}
class Node:
    """
    gameState: chess.Board
    fen: str
    visited: int # how many times this node has been visited N(s)
    totalValue: float # total value of this node W(s)
    averageValue: float # average value of this node W(s)/N(s) - Q(s)
    children: dict  # children nodes
    """

    def __init__(self, gameState: chess.Board):
        self.gameState = gameState
        fens = gameState.fen().split(" ")[:2]
        self.ffen = gameState.fen()
        self.fen = " ".join(fens)
        self.visited = 0
        self.totalValue = 0.0
        self.averageValue = 0.0
        self.children = {}

    def update(self, value: float):
        # print(f"UPDATING {self.fen}")
        self.visited += 1
        self.totalValue += value
        self.averageValue = self.totalValue / self.visited

    def expand(self):
        if not self.children:
            for move in self.gameState.legal_moves:
                newGameState = self.gameState.copy()
                newGameState.push(move)
                self.children[move] = get_node(newGameState)

    def UCT(self, parentVisited: int, c: float = math.sqrt(2)) -> float:
        if self.visited == 0:
            return float("inf")
        """Upper Confidence Bound for Trees (UCT)"""
        """Q(s, a) + c * sqrt(ln(N(s)) / N(s, a))"""
        """Q(s,a) = average win rate for move a"""
        """N(s) = number of times parent node has been visited"""
        """N(s,a) = number of times child node has been visited"""
        """c = exploration parameter, 1.41"""
        parent_visits = max(1, parentVisited)
        return self.averageValue + (
            c * math.sqrt(math.log(parent_visits) / self.visited)
        )


  
def get_node(gamestate: chess.Board) -> Node:
    fens = gamestate.fen().split(" ")[:2]
    fen = " ".join(fens)
    if fen in TranspositionTable:
        return TranspositionTable[fen]
    TranspositionTable[fen] = Node(gamestate)
    return TranspositionTable[fen]
    
def build(iters=iters):
    board = chess.Board()
    path = [board.fen()]
    
    with open('dicto.pkl', 'rb') as file:
            TranspositionTable = pickle.load(file)

    for _ in range(iters):
        while not board.is_game_over():
            moves=list(board.legal_moves)
            caps_checks = [m for m in moves if (board.is_capture(m) or board.gives_check(m))]
            move = random.choice(caps_checks) if caps_checks else random.choice(moves)
            board.push(move)
            path.append(board.fen())
        
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                res = 1.0
            elif result == "0-1":
                res = -1.0
            else:
                res =  0.0
    
        for i in reversed(path):
            board = chess.Board(i)
            node = get_node(board)
            print(f"{node.ffen}")
            node.update(res)
            res = -res

        board = chess.Board()
        path= [board.fen()]
        iters -= 1

    with open('dicto.pkl', 'wb') as file:
        pickle.dump(TranspositionTable, file)
        # Use pickle.dump to serialize and save the object to the file

####################
# DEMO FUNCTIONS
####################
def runStats():
      with open('dicto.pkl', 'rb') as file:
        table = pickle.load(file)
      max_node = max(table.values(), key=lambda n: getattr(n, "totalValue", float("-inf")))
      min_node = min(table.values(), key=lambda n: getattr(n, "totalValue", float("inf")))
      high_avg = max(table.values(), key=lambda n: getattr(n, "averageValue", float("-inf")))
      low_avg = min(table.values(), key=lambda n: getattr(n, "averageValue", float("inf")))
      most_visited = max(table.values(), key=lambda n: getattr(n, "visited", float("inf")))

      print(f"MAX: {max_node.fen} ({max_node.totalValue}), VISITS: {max_node.visited}, AVG: {max_node.averageValue}")
      print(f"LOWEST: {min_node.fen} ({min_node.totalValue}), VISITS: {min_node.visited}, AVG: {min_node.averageValue}")
      print(f"Most Visisted: {most_visited.fen} ({most_visited.totalValue}), VISITS: {most_visited.visited}, AVG: {most_visited.averageValue}")
      print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
      print(f"T TABLE LEN: {len(table.keys())}")
      print(f"TABLE SIZE: {sys.getsizeof(table)/1000000} mb")
      
def graphit(path=None):
    with open('dicto.pkl', 'rb') as file:
        treedict = pickle.load(file)
    nodes = list(treedict.values())

    # safe attribute extraction with fallbacks
    visits = [getattr(n, "visited", 0) for n in nodes]
    positions = [getattr(n, "fen", getattr(n, "fen", "")) for n in nodes]
    scores = [getattr(n, "averageValue", 0.0) for n in nodes]

    x = list(range(len(positions)))
    y = list(scores) 

    sizes = [max(10, v) for v in visits]
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_facecolor("black")
    sc = ax.scatter(x, y, s=sizes, c=scores, cmap="RdPu", alpha=0.8, linewidths=0.2)
        #TODO: ADD ABILITY TO PASS IN LIST AND VISUALIZE A PATH, needs to get index for all moves but this is ok. Will have to account for novelties
    if path:
        xVals=[positions.index(a) for a in path]
        yVals=[getattr(n, "averageValue", 0.0) for n in path]
        plt.plot(xVals, yVals, color='red')

    cbar = plt.colorbar(sc, ax=ax, label="averageValue")
    # ensure colorbar and ticks are readable on black background
    cbar.ax.yaxis.label.set_color("black")
    cbar.ax.tick_params(colors="black")

    ax.set_title(f"Positions: {len(positions)}", color="black")
    plt.show()

##################
#UTIL FUNCTIONS
##################
def sendpos(fen: str):
    payload = {
        "fen": fen,
        "evalSource1": "N/A",
        "evalSource2": "N/A"
    }
    response = requests.post(uri, json=payload)  # Use json=payload
    print(response.status_code, response.text)
    print(f"pos sent! {response}")

async def query_stockfish(fen: str):
    async with websockets.connect(uri) as ws:
        # Send position and go
        await ws.send(json.dumps({"type": "uci:command", "payload": f"position fen {fen}"}))
        await ws.send(json.dumps({"type": "uci:command", "payload": "go depth 2"}))
        # Wait for bestmove response
        async for msg in ws:
            print(msg)
            data = json.loads(msg)
            if data.get("type") == "uci:response" and ("bestmove" in data.get("payload")):
                return data["payload"].split(" ")[1]

def get_bestmove(fen: str):
    res = asyncio.run(query_stockfish(fen))
    print(f"RES: {res}")
    return res

def play_game(iters=17):
    board = chess.Board()
    path = [board.fen()]
    
    with open('dicto.pkl', 'rb') as file:
            TranspositionTable = pickle.load(file)

    for _ in range(iters):
        if random.random() <= 0.5:
            fish = chess.BLACK
        else:
            fish = chess.WHITE
        while not board.is_game_over():
            if board.turn == fish:
                move = chess.Move.from_uci(get_bestmove(board.fen()))
                board.push(move)
                path.append(board.fen())
                sendpos(board.fen())
                time.sleep(.7)
            else:
                node = get_node(board)
                if not node.children:
                   node.expand()
                
                moves=list(board.legal_moves)
                caps_checks = [m for m in moves if (board.is_capture(m) or board.gives_check(m))]
                move = random.choice(caps_checks) if caps_checks else random.choice(moves)

                board.push(move)
                path.append(board.fen())
                sendpos(board.fen())
                time.sleep(.1)
        
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                res = 1.0
            elif result == "0-1":
                res = -1.0
            else:
                res =  0.0
    
        for i in reversed(path):
            board = chess.Board(i)
            node = get_node(board)
            print(f"{node.ffen}")
            node.update(res)
            res = -res

        board = chess.Board()
        path= [board.fen()]
        iters -= 1

        with open('dicto.pkl', 'wb') as file:
            pickle.dump(TranspositionTable, file)
            print("DICTO SAVED")
            print("DICTO SAVED")
            print("DICTO SAVED")
            # Use pickle.dump to serialize and save the object to the file
        print("Game over!~")

# build(iters=5413)
play_game()
runStats()
graphit()