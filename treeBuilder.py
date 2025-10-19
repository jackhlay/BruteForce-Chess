import chess
import math
import random
import redis
import pickle
import sys


iters = 537
red = redis.Redis(host='localhost', port=6379, db=0)
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
    
def update_table(redis_client=red):
    raw = redis_client.get("T_TABLE")
    if not raw:
        return

    try:
        storedTable = pickle.loads(raw)
        if not isinstance(storedTable, dict):
            return
        TranspositionTable.update(storedTable)
    except Exception as e:
        print(f"Failed to load transposition table: {e}")

def build(iters=iters):
    board = chess.Board()
    path = [board]
    
    if TranspositionTable == {}:
        update_table()
        # print(TranspositionTable.keys())

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
            print(f"UPDATING {i}, TYPE:({type(i)})")
            if type(i) == type(chess.Board()):
                board = i
            else:
                board = chess.Board(i)
            node = get_node(board)
            print(f"{node.ffen}")
            node.update(res)
            res = -res

        board = chess.Board()
        path= [board]
        iters -= 1

    print(f"T TABLE LEN: {len(TranspositionTable.keys())})")
    red.set("T_TABLE", pickle.dumps(TranspositionTable))

def runStats():
      raw = red.get("T_TABLE")
      table = pickle.loads(raw)
      max_node = max(table.values(), key=lambda n: getattr(n, "totalValue", float("-inf")))
      min_node = min(table.values(), key=lambda n: getattr(n, "totalValue", float("inf")))
      high_avg = max(table.values(), key=lambda n: getattr(n, "averageValue", float("-inf")))
      low_avg = min(table.values(), key=lambda n: getattr(n, "averageValue", float("inf")))
      most_visited = max(table.values(), key=lambda n: getattr(n, "visited", float("inf")))

      print(f"MAX: {max_node.fen} ({max_node.totalValue}), VISITS: {max_node.visited}, AVG: {max_node.averageValue}")
      print(f"LOWEST: {min_node.fen} ({min_node.totalValue}), VISITS: {min_node.visited}, AVG: {min_node.averageValue}")
    #   print(f"Highest Average: {high_avg.fen} ({high_avg.totalValue}), VISITS: {high_avg.visited}")
    #   print(f"Lowest Average: {low_avg.fen} ({low_avg.totalValue}), VISITS: {low_avg.visited}")
      print(f"Most Visisted: {most_visited.fen} ({most_visited.totalValue}), VISITS: {most_visited.visited}, AVG: {most_visited.averageValue}")
      print(f"TABLE SIZE: {sys.getsizeof(table)/1000000} mb")
      


build(iters=2500)
runStats()