import bulletchess as bc
import bulletchess.utils as bcu
import sys
import time

# ─────────────────────────────────────────────────────────────
# TUNING CONSTANTS
# ─────────────────────────────────────────────────────────────
PIECE_VAL = {
    "pawn":   100,
    "knight": 320,
    "bishop": 330,
    "rook":   500,
    "queen":  900,
    "king":   20000,
}
 
FUTILITY_MARGIN = [0, 150, 300, 500]   # per depth 0-3, in centipawns
DELTA_MARGIN    = 200                   # q-search delta pruning
ASP_WINDOW      = 50                    # centipawns
LMR_MIN_DEPTH   = 3                     # don't reduce at shallow depth
LMR_MIN_MOVE    = 4                     # start reducing after this many moves tried
MAX_KILLERS     = 2
INF             = 10_000_000
 
# ─────────────────────────────────────────────────────────────
# TRANSPOSITION TABLE
# ─────────────────────────────────────────────────────────────
TT_EXACT = 0
TT_LOWER = 1    # fail-high: score is a lower bound
TT_UPPER = 2    # fail-low:  score is an upper bound
 
_tt = {}        # key → (depth, score, flag, best_move)
 
def tt_probe(key, depth, alpha, beta):
    entry = _tt.get(key)
    if entry is None:
        return None, None
    stored_depth, score, flag, best_move = entry
    if stored_depth >= depth:
        if flag == TT_EXACT:
            return score, best_move
        if flag == TT_LOWER and score >= beta:
            return score, best_move
        if flag == TT_UPPER and score <= alpha:
            return score, best_move
    # Not usable for cutoff but still return best_move for ordering
    return None, best_move
 
def tt_store(key, depth, score, flag, best_move):
    existing = _tt.get(key)
    if existing is None or existing[0] < depth:
        _tt[key] = (depth, score, flag, best_move)
 
def tt_clear():
    _tt.clear()

def tt_size():
    return sys.getsizeof(_tt) / 1e6, len(_tt)
 
# ─────────────────────────────────────────────────────────────
# HEURISTICS
# ─────────────────────────────────────────────────────────────
_killers = [[None] * MAX_KILLERS for _ in range(256)]
_history = {}
 
def _clear_heuristics():
    global _killers, _history
    _killers = [[None] * MAX_KILLERS for _ in range(256)]
    _history = {}
 
def _store_killer(move, ply):
    if move not in _killers[ply]:
        _killers[ply] = [move] + _killers[ply][:MAX_KILLERS - 1]
 
def _update_history(move, depth):
    k = (move.origin, move.destination)
    _history[k] = _history.get(k, 0) + depth * depth
 
# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────
def _popcount(bb):
    return bin(int(bb)).count("1")
 
def static_eval(board):
    """
    Material score from the perspective of the side to move.
    Positive = good for the side to move.
    """
    turn = 1 if board.turn == bc.WHITE else -1
    wb = bcu.white_bitboard(board)
    bb_ = bcu.black_bitboard(board)
    pb = bcu.pawn_bitboard(board)
    nb = bcu.knight_bitboard(board)
    bib = bcu.bishop_bitboard(board)
    rb = bcu.rook_bitboard(board)
    qb = bcu.queen_bitboard(board)
 
    score = (
        PIECE_VAL["pawn"] * (_popcount(wb & pb)  - _popcount(bb_ & pb))  +
        PIECE_VAL["knight"] * (_popcount(wb & nb)  - _popcount(bb_ & nb))  +
        PIECE_VAL["bishop"] * (_popcount(wb & bib) - _popcount(bb_ & bib)) +
        PIECE_VAL["rook"] * (_popcount(wb & rb)  - _popcount(bb_ & rb))  +
        PIECE_VAL["queen"] * (_popcount(wb & qb)  - _popcount(bb_ & qb))
    )
    return turn * score
 
# ─────────────────────────────────────────────────────────────
# MOVE ORDERING
# ─────────────────────────────────────────────────────────────
def _piece_val(board: bc.Board, sq: bc.Square):
    piece = board[sq]
    if piece is None:
        return 0
    return PIECE_VAL.get(str(piece.piece_type).lower(), 0)
 
def _score_move(board: bc.Board, m: bc.Move, ply, tt_move):
    """Score a single move for ordering (higher = search first)."""
    if tt_move is not None and m == tt_move:
        return 20_000_000
 
    if m.is_promotion:
        return 15_000_000 + PIECE_VAL.get(str(m.promotion).lower(), 0)
 
    if m.is_capture(board):
        victim   = _piece_val(board, m.destination)
        attacker = _piece_val(board, m.origin)
        return 10_000_000 + victim * 10 - attacker
 
    if m in _killers[ply]:
        return 9_000_000 - _killers[ply].index(m)
 
    return _history.get((m.origin, m.destination), 0)
 
def order_moves(board, ply, tt_move=None):
    moves = list(board.legal_moves())
    moves.sort(key=lambda m: _score_move(board, m, ply, tt_move), reverse=True)
    return moves
 
def order_captures(board):
    caps = [m for m in board.legal_moves() if m.is_capture(board)]
    caps.sort(
        key=lambda m: _piece_val(board, m.destination) * 10 - _piece_val(board, m.origin),
        reverse=True
    )
    return caps
 
# ─────────────────────────────────────────────────────────────
# REPETITION / DRAW
# ─────────────────────────────────────────────────────────────
_pos_history = []
 
def _is_repetition(board: bc.Board):
    key = board.__hash__()
    return _pos_history.count(key) >= 2
 
# ─────────────────────────────────────────────────────────────
# QUIESCENCE SEARCH
# ─────────────────────────────────────────────────────────────
_nodes = 0
 
def quiescence(board: bc.Board, alpha, beta):
    global _nodes
    _nodes += 1
 
    stand_pat = static_eval(board)
 
    if stand_pat >= beta:
        return beta
    if stand_pat + 900 + DELTA_MARGIN < alpha:  # can't possibly raise alpha
        return alpha
    alpha = max(alpha, stand_pat)
 
    for m in order_captures(board):
        # Per-capture delta pruning
        if stand_pat + _piece_val(board, m.destination) + DELTA_MARGIN < alpha:
            continue
 
        board.apply(m)
        score = -quiescence(board, -beta, -alpha)
        board.undo()
 
        if score >= beta:
            return beta
        alpha = max(alpha, score)
 
    return alpha
 
# ─────────────────────────────────────────────────────────────
# ALPHA-BETA
# ─────────────────────────────────────────────────────────────
_stop_flag  = False
_root_depth = 1   # set at the start of each depth iteration in find_best_move
 
def alphabeta(board: bc.Board, depth, alpha, beta, ply, deadline):
    global _nodes, _stop_flag, _root_depth
 
    if _stop_flag or time.perf_counter() >= deadline:
        _stop_flag = True
        return 0
 
    # Draw checks
    if _is_repetition(board):
        return 0
    if board in bc.DRAW:
        return 0
 
    # Mate distance pruning
    mating_score = INF - ply
    if alpha >= mating_score:
        return alpha
 
    # TT probe
    key = board.__hash__()
    tt_score, tt_move = tt_probe(key, depth, alpha, beta)
    if tt_score is not None and ply > 0:
        return tt_score
 
    # Check extension — capped so depth can never grow beyond root depth.
    # Without the cap, a chain of checks increments depth every call
    # instead of decrementing it, so `depth <= 0` is never reached and
    # the search recurses to thousands of plies.
    # The cap allows at most 1 extension per 2 plies of root depth,
    # which is generous but finite.
    in_check = board in bc.CHECK
    if in_check and ply < _root_depth * 2:
        depth += 1
 
    # Drop into quiescence and leaves
    if depth <= 0:
        return quiescence(board, alpha, beta)
 
    # Check for legal moves
    moves = order_moves(board, ply, tt_move)
    if not moves:
        return -(INF - ply) if in_check else 0  # checkmate or stalemate
 
    # Futility pruning setup (only at low depths, not in check)
    do_futility = (
        not in_check and
        depth <= 3 and
        depth >= 1 and
        abs(alpha) < INF // 2 and
        abs(beta)  < INF // 2
    )
    futility_base = (static_eval(board) + FUTILITY_MARGIN[depth]) if do_futility else None
 
    original_alpha = alpha
    best_score = -INF
    best_move  = None
 
    for i, m in enumerate(moves):
        is_cap   = m.is_capture(board)
        is_promo = m.is_promotion
 
        # Futility pruning: skip quiet moves that can't improve alpha
        if do_futility and not is_cap and not is_promo:
            if futility_base < alpha:
                if best_score > -INF:   # only prune if we have at least one move scored
                    continue
 
        # LMR: reduce depth for late, quiet moves
        reduction = 0
        if (
            depth >= LMR_MIN_DEPTH and
            i >= LMR_MIN_MOVE and
            not is_cap and
            not is_promo and
            not in_check and
            m not in _killers[ply]
        ):
            reduction = 1   # flat 1-ply reduction; safe and well-tested
 
        new_depth = depth - 1 - reduction
 
        board.apply(m)
        _nodes += 1
        _pos_history.append(board.__hash__())
 
        # PVS: full window for first move, null window + re-search for rest
        if i == 0:
            score = -alphabeta(board, new_depth, -beta, -alpha, ply + 1, deadline)
        else:
            score = -alphabeta(board, new_depth, -alpha - 1, -alpha, ply + 1, deadline)
            # Re-search at full depth if it beat alpha (LMR might have missed something)
            if not _stop_flag and score > alpha and (score < beta or reduction > 0):
                score = -alphabeta(board, depth - 1, -beta, -alpha, ply + 1, deadline)
 
        _pos_history.pop()
        board.undo()
 
        if _stop_flag:
            return 0
 
        if score > best_score:
            best_score = score
            best_move  = m
 
        if score >= beta:
            if not is_cap:
                _store_killer(m, ply)
                _update_history(m, depth)
            tt_store(key, depth, score, TT_LOWER, best_move)
            return score   # fail-hard: return actual score, not beta
 
        alpha = max(alpha, score)
 
    # Store result
    flag = TT_UPPER if best_score <= original_alpha else TT_EXACT
    tt_store(key, depth, best_score, flag, best_move)
    return best_score
 
# ─────────────────────────────────────────────────────────────
# ROOT / ITERATIVE DEEPENING
# ─────────────────────────────────────────────────────────────
def find_best_move(board:bc.Board, time_limit=5.0, max_depth=64):
    """
    Main entry point. Call this each turn.
 
    find_best_move(board, time_limit=3.0)  →  bc.Move
 
    The TT is NOT cleared between calls so it benefits from
    previous search results (hash move ordering is much better).
    Call tt_clear() manually if you start a new game.
    """
    global _nodes, _stop_flag, _root_depth
 
    _clear_heuristics()
    _stop_flag   = False
    _nodes       = 0
 
    deadline   = time.perf_counter() + time_limit
    best_move  = list(board.legal_moves())[0]
    prev_score = 0
 
    for depth in range(1, max_depth + 1):
        if time.perf_counter() >= deadline:
            break
 
        _root_depth = depth   # cap check extensions relative to this
 
        # Aspiration windows: skip for first few depths (scores too volatile)
        if depth <= 3:
            alpha, beta = -INF, INF
        else:
            alpha = prev_score - ASP_WINDOW
            beta  = prev_score + ASP_WINDOW
 
        # Aspiration retry loop
        while True:
            iter_best_score = -INF
            iter_best_move  = None
            root_alpha      = alpha  # remember where we started for failure check
 
            tt_move = (_tt.get(board.__hash__()) or (None,) * 4)[3]
            ordered = order_moves(board, 0, tt_move)
 
            for m in ordered:
                if _stop_flag or time.perf_counter() >= deadline:
                    _stop_flag = True
                    break
 
                board.apply(m)
                _nodes += 1
                _pos_history.append(board.__hash__())
                score = -alphabeta(board, depth - 1, -beta, -alpha, 1, deadline)
                _pos_history.pop()
                board.undo()
 
                if score > iter_best_score:
                    iter_best_score = score
                    iter_best_move  = m
 
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    break
 
            if _stop_flag:
                break
            if iter_best_score <= root_alpha - ASP_WINDOW:
                alpha = -INF                # fail low → open left side fully
            elif iter_best_score >= beta:
                beta = INF                  # fail high → open right side fully
            else:
                break                       # score landed inside window, done
 
        if not _stop_flag and iter_best_move is not None:
            best_move  = iter_best_move
            prev_score = iter_best_score
 
        elapsed = time.perf_counter() - (deadline - time_limit)
        nps = int(_nodes / max(elapsed, 1e-9))
        print(
            f"depth={depth:2d}  score={prev_score:+6d}cp  "
            f"nodes={_nodes:,}  nps={nps:,}  "
            f"best={best_move}  t={elapsed:.2f}s"
        )
 
        if _stop_flag:
            break
        if abs(prev_score) > INF // 2:
            print("Forced mate found, stopping early.")
            break
 
    return best_move
 
 
# ─────────────────────────────────────────────────────────────
# USAGE
# ─────────────────────────────────────────────────────────────
#
#   At game start:
#       tt_clear()
#
#   Each move:
#       move = find_best_move(board, time_limit=3.0)
#       board.apply(move)