import chess
import math
import random

TranspositionTable = {}

class Node:

    """
    gameState: chess.Board
    fen: str
    move: str
    visited: int # how many times this node has been visited N(s)
    totalValue: float # total value of this node W(s)
    averageValue: float # average value of this node W(s)/N(s) - Q(s)
    parent: 'Node' # parent node
    children: dict  # children nodes
    """

    def __init__(self, gameState: chess.Board, move: chess.Move = None, parent: 'Node' = None):
        self.gameState = gameState
        self.fen = gameState.fen()
        self.move = move
        self.visited = 0
        self.totalValue = 0.0
        self.averageValue = 0.0
        self.parent = parent
        self.children = {}

    def update(self, value: float):
        self.visited += 1
        self.totalValue += value
        self.averageValue = self.totalValue / self.visited

    def expand(self):
        if not self.children:
            for move in self.gameState.legal_moves:
                newGameState = self.gameState.copy()
                newGameState.push(move)
                self.children[move] = get_node(newGameState, move, parent=self)

    def UCT(self, c: float = math.sqrt(2)) -> float:
        if self.visited == 0:
            return float('inf')
        """Upper Confidence Bound for Trees (UCT)"""
        '''Q(s, a) + c * sqrt(ln(N(s)) / N(s, a))'''
        '''Q(s,a) = average win rate for move a'''
        '''N(s) = number of times parent node has been visited'''
        '''N(s,a) = number of times child node has been visited'''
        '''c = exploration parameter, 1.41'''
        parent_visits = max(1, self.parent.visited) if self.parent is not None else 1
        return self.averageValue + (c * math.sqrt(math.log(parent_visits) / self.visited))

def get_node(gamestate: chess.Board, move:chess.Move = None, parent: Node = None) -> Node:
    fen = gamestate.fen()
    if fen in TranspositionTable:
        return TranspositionTable[fen]
    TranspositionTable[fen] = Node(gamestate, move, parent)
    return TranspositionTable[fen] 

def select_with_path(root: Node):
    current = root
    path = [current]
    
    while current.children:
        current = max(current.children.values(), key=lambda n: n.UCT())
        path.append(current)
    return current, path

def expand_one(node: Node) -> Node:
    node.expand()
    if node.children:
        return random.choice(list(node.children.values()))
    return node

def rollout(board: chess.Board, max_deptht: int=19) -> float:
    depth = 0
    while not board.is_game_over() and depth < max_deptht:
        moves = board.legal_moves
        #
        caps_checks = [m for m in moves if (board.is_capture(m) or board.gives_check(m))]
        # move = random.choice(list(caps_checks) if caps_checks else list(moves))
        move = random.choice(list(moves))
        board.push(move)
        depth += 1
    
    if board.is_game_over():
        result = board.result()
        if result == '1-0': return 1.0
        elif result == '0-1': return -1.0
        else: return 0.0
    
    # TranspositionTable.clear()
    return heurEval(board)

def backProp_Path(path , valueFromWhite: float):
   for node in reversed(path):
       signed = valueFromWhite if node.gameState.turn == chess.WHITE else -valueFromWhite
       node.update(value=signed)

def mcts(root: Node, iterations: int = 1000) -> chess.Move:
    for i in range(iterations):
        # print(f"ITER NUMBER {i}")
        leaf,path = select_with_path(root)
        child = expand_one(leaf)
        if child is not path[-1]:
            path.append(child)
        board_copy = child.gameState.copy()
        value = rollout(board_copy)
        backProp_Path(path, value)
    
    best_move = max(root.children.values(), key=lambda n: n.visited).move
    return best_move

def heurEval(board: chess.Board) -> float:
    """A simple heuristic evaluation function based on material count."""
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
}
    
    value = 0
    for piece_type, pv in piece_values.items():
        value += len(board.pieces(piece_type, chess.WHITE)) * pv
        value -= len(board.pieces(piece_type, chess.BLACK)) * pv
    if board.turn == chess.BLACK: 
        blackMoves = len(list(board.legal_moves))
        WBOARD = board.push(chess.Move.null())
        whiteMoves = len(list(board.legal_moves))
        board.push(chess.Move.null())

    else:
        whiteMoves = len(list(board.legal_moves))
        board.push(chess.Move.null())
        blackMoves = len(list(board.legal_moves))
        board.push(chess.Move.null())

    return value / 100.0 + ((whiteMoves-blackMoves)/10)

if __name__ == "__main__":
    board = chess.Board()
    root = get_node(board)
    best_move = mcts(root, iterations=1300)
    print(f"Best move: {best_move}")