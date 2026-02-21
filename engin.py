import berserk
import bulletchess as bc
import bulletchess.utils as bcu

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

node_count = 0

def isTerminal(board):
    terminal_statuses = [bcu.MATE, bcu.CHECKMATE, bcu.DRAW ]

    for s in terminal_statuses:
        if board in s:
            return True
    return False

def scoreMoves(move: bc.Move, board: bc.Board):
    if move.is_capture(board):
        pieceVal = { "p" : 100, "k":300, "b":320, "r":600, "q":900} 
        
        origSq = bc.Square.from_str(move.origin)
        destSq = bc.Square.from_str(move.destination)

        origPiece = str(board[origSq]).lower()
        destPiece = str(board[destSq]).lower()
        

        return((pieceVal[destPiece]*10) - (pieceVal[origPiece])) #(dest square piece value * 5) - (orig square piece value)
    return 0

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

def negaMax(board: bc.Board, depth, alpha, beta):
    global node_count

    if depth == 0:
        return quiesce(board, alpha, beta)
    
    moves = list(board.legal_moves())
    moves.sort(key=lambda m: scoreMoves(m, board), reverse=True)

    for m in moves:
        board.apply(m)
        score = -negaMax(board, depth-1, -beta, -alpha)
        node_count += 1
        board.undo()
        if score >= beta:
           return beta
        if score > alpha:
            alpha = score

    return alpha


def ABPrune(board: bc.Board, depth, alpha, beta, maximizing, reset=False):
    global node_count
    
    if depth == 0:
        return quiesce(board, alpha, beta)
    
    if isTerminal(board):
        return bcu.evaluate(board)
    
    if maximizing:
        maxEval = float("-inf")
        for m in board.legal_moves():
            board.apply(m)
            val = ABPrune(board, depth-1, alpha, beta, maximizing=False)
            node_count += 1
            board.undo()
            maxEval = max(maxEval, val)
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return maxEval
    else:
        minEval = float("inf")
        for m in board.legal_moves():
            board.apply(m)
            val = ABPrune(board, depth-1, alpha, beta, maximizing=True)
            node_count += 1
            board.undo()
            minEval = min(minEval, val)
            beta = min(beta, val)
            if beta <= alpha:
                break
        return minEval

def rootEval(board: bc.Board, depth):

    bestMove = None

    alpha = float("-inf")
    beta = float("inf")

    for m in board.legal_moves:
        board.apply(m)
        score = -negaMax(board, depth-1, -beta, -alpha)
        board.undo()

        if score > alpha:
            alpha =score
            bestMove = m

    return bestMove



    # get bitboards the active side
    # move ordering for each 
    # use the destinations for each move in addition to bitboard?
    # ...
    # profit