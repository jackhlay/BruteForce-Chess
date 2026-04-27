"""
Required env:
    LICHESS_TOKEN
 
Optional env:
    TIME_LIMIT    seconds per move (default: 3.247)
    MAX_DEPTH     max search depth  (default: 64)
"""
 
import os
import sys
import time
import logging
import random
import threading
 
import berserk
from bulletchess import *
 
from engin import find_best_move, tt_clear, _pos_history
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bot")

TOKEN      = os.environ["LICHESS_TOKEN"]
CONCURRENCY = int(os.environ.get("CONCURRENCY", 3))
TIME_LIMIT = float(os.environ.get("TIME_LIMIT", 3.247))
MAX_DEPTH  = int(os.environ.get("MAX_DEPTH", 999))

 
active_games: set[str] = set()
games_lock = threading.Lock()

session = berserk.TokenSession(TOKEN)
client  = berserk.Client(session=session)
 
 
def board_from_moves(move_list: list[str]) -> Board:
    """Replay a UCI move list and return the resulting board.
    Seeds _pos_history so the engine detects repetitions against real game history."""
    board = Board()
    _pos_history.clear()
    _pos_history.append(board.__hash__())
    for uci in move_list:
        board.apply(Move.from_uci(uci))
        _pos_history.append(board.__hash__())
    return board
 
 
def play_game(game_id: str, bot_color: str):
    log.info(f"[{game_id}] playing as {bot_color}")
    tt_clear()
 
    for event in client.bots.stream_game_state(game_id):
        etype = event.get("type")
 
        if etype == "gameState" or etype == "gameFull":
            state = event["state"] if etype == "gameFull" else event
            if state.get("status") not in ("started", "created"):
                log.info(f"[{game_id}] game over: {state.get('status')}")
                with games_lock:
                    active_games.remove(game_id)
                return
 
            moves    = [m for m in state.get("moves", "").split() if m]
            our_turn = (len(moves) % 2 == 0) == (bot_color == "white")
            if not our_turn:
                continue
 
            board = board_from_moves(moves)
            log.info(f"[{game_id}] thinking (ply {len(moves) + 1})...")
            if len(moves)+1 < 6:
                move = random.choice(board.legal_moves())
            else:
                player_time = state["wtime"] if bot_color == "white" else state["btime"]
                log.info(f"PLAYER TIME: {player_time}")
                timeEst = .04 * (player_time) 
                log.info(f"TIME COMPARISON: HARD LIMIT: {TIME_LIMIT}, ESTIMATED (~4%): {timeEst.seconds}")
                move = find_best_move(board, time_limit=TIME_LIMIT, max_depth=MAX_DEPTH)
            log.info(f"[{game_id}] playing {move}")
            client.bots.make_move(game_id, str(move))
 
 
def handle_events():
    bot_id = client.account.get()["id"]
    log.info(f"logged in as @{bot_id}")
 
    for event in client.bots.stream_incoming_events():
        etype = event.get("type")
        if len(active_games) >= CONCURRENCY:
                client.bots.decline_challenge(cid)
                log.info("DECLINED GAME {}, 3 RUNNING ALRADY", chal["id"])
 
        if etype == "challenge":
            chal = event["challenge"]
            cid  = chal["id"]

            isNotStandard = chal.get("variant", {}).get("key") != "standard"
            isRated = chal.get("rated")
            isCorrespondence = chal.get('speed') == "correspondence"
            atCapacity = len(active_games) >= CONCURRENCY

            if isNotStandard or isRated or isCorrespondence or atCapacity:
                client.bots.decline_challenge(cid)
                log.info(f"DECLINED GAME {cid}, NonStandard?: {isNotStandard}, RATED?: {isRated}, Correspondence?:{isCorrespondence}, At CAPACITY?: {atCapacity}")
            else:
                log.info(f"accepting challenge {cid}")

                client.bots.accept_challenge(cid)
 
        elif etype == "gameStart":
            game      = event["game"]
            game_id   = game["gameId"]
            bot_color = game["color"]
            with games_lock:
                active_games.add(game_id)
            threading.Thread(
                target=play_game,
                args=(game_id, bot_color),
                daemon=True,
            ).start()
 
 
if __name__ == "__main__":
    me = client.account.get()
    if me.get("title") != "BOT":
        log.error("Account is not a BOT. Run: POST /api/bot/account/upgrade")
        sys.exit(1)
    print("BOT STATUS CONFIRMED")
    while True:
        try:
            handle_events()
        except Exception as e:
            log.exception(f"event loop error: {e}")
            time.sleep(5)
            session = berserk.TokenSession(TOKEN)
            client  = berserk.Client(session=session)
            continue
