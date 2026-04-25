import tkinter as tk
from tkinter import simpledialog
import chess
import bulletchess as bc
from engin import find_best_move
import random
import threading

class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Boby Chess v1")
        self.board = chess.Board()
        self.board_bc = bc.Board()
        self.canvas_size = 973
        self.search_time = .08
        self.engine_move_pending = False
        self.selected_square = None
        self.dragging = False
        self.drag_start = None
        
        # Game setup
        self.white_player = tk.StringVar(value="human")
        self.black_player = tk.StringVar(value="cpu")
        
        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Side menu
        menu_frame = tk.Frame(main_frame, bg="#34495E", width=180)
        menu_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Title
        title = tk.Label(menu_frame, text="Chess", font=("Arial", 16, "bold"), 
                        bg="#34495E", fg="white")
        title.pack(pady=10)
        
        # White player
        tk.Label(menu_frame, text="White:", font=("Arial", 11, "bold"),
                bg="#34495E", fg="white").pack(anchor=tk.W, padx=10, pady=(10, 5))
        tk.Radiobutton(menu_frame, text="Human", variable=self.white_player, 
                      value="human", bg="#34495E", fg="white", selectcolor="#34495E").pack(anchor=tk.W, padx=20)
        tk.Radiobutton(menu_frame, text="CPU", variable=self.white_player, 
                      value="cpu", bg="#34495E", fg="white", selectcolor="#34495E").pack(anchor=tk.W, padx=20)
        
        # Black player
        tk.Label(menu_frame, text="Black:", font=("Arial", 11, "bold"),
                bg="#34495E", fg="white").pack(anchor=tk.W, padx=10, pady=(10, 5))
        tk.Radiobutton(menu_frame, text="Human", variable=self.black_player, 
                      value="human", bg="#34495E", fg="white", selectcolor="#34495E").pack(anchor=tk.W, padx=20)
        tk.Radiobutton(menu_frame, text="CPU", variable=self.black_player, 
                      value="cpu", bg="#34495E", fg="white", selectcolor="#34495E").pack(anchor=tk.W, padx=20)
        
        # Separator
        tk.Frame(menu_frame, bg="white", height=2).pack(fill=tk.X, pady=10, padx=10)
        
        # Buttons
        tk.Button(menu_frame, text="Reset", command=self.reset,
                 bg="#34495E", fg="white", width=15).pack(pady=5)
        tk.Button(menu_frame, text="Load FEN", command=self.load_fen,
                 bg="#34495E", fg="white", width=15).pack(pady=5)
        
        # Canvas
        self.canvas = tk.Canvas(main_frame, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        self.draw_board()
    
    def on_press(self, event):
        if not self.can_move():
            return
        square_size = self.canvas_size // 8
        col = event.x // square_size
        row = event.y // square_size
        self.selected_square = chess.Square((7 - row) * 8 + col)
        self.dragging = True
        self.drag_start = (event.x, event.y)
    
    def on_drag(self, event):
        if self.dragging:
            self.draw_board()
            square_size = self.canvas_size // 8
            piece = self.board.piece_at(self.selected_square)
            if piece:
                piece_unicode = self.piece_to_unicode(piece.symbol())
                text_color = "white" if piece.color else "black"
                self.canvas.create_text(event.x, event.y, text=piece_unicode, 
                                       font=("Arial", 50), fill=text_color)
    
    def on_release(self, event):
        if not self.dragging:
            return
        
        self.dragging = False
        square_size = self.canvas_size // 8
        col = event.x // square_size
        row = event.y // square_size
        to_square = chess.Square((7 - row) * 8 + col)
        
        if to_square != self.selected_square:
            move = chess.Move(self.selected_square, to_square)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.board_bc.apply(bc.Move.from_uci(move.uci()))
                self.draw_board()
                self.root.after(500, self.engine_move)
        
        self.selected_square = None
        self.draw_board()
    
    def engine_move(self):
        """Make engine move if it's the engine's turn"""
        if self.board.is_game_over() or self.engine_move_pending:
            return
        
        white_is_cpu = self.white_player.get() == "cpu"
        black_is_cpu = self.black_player.get() == "cpu"
        
        try:
            if self.board.fullmove_number < 7  and not self.can_move():
                best_move = random.choice(self.board_bc.legal_moves())
                self.board_bc.apply(best_move)
                self.board.push(chess.Move.from_uci(best_move.uci()))
                self.draw_board()
                self.root.update()
                self.root.after(30, self.engine_move)

            elif self.board.turn == chess.WHITE and white_is_cpu:
                self.engine_move_pending = True
                thread = threading.Thread(target=self._search_and_move, daemon=True)
                thread.start()
                # best_move = find_best_move(self.board_bc, self.search_time, 999)
                # self.board.push(chess.Move.from_uci(best_move.uci()))
                # self.board_bc.apply(best_move)
                # self.draw_board()
                # self.root.update()
                # self.root.after(30, self.engine_move)

            elif self.board.turn == chess.BLACK and black_is_cpu:
                self.engine_move_pending = True
                thread = threading.Thread(target=self._search_and_move, daemon=True)
                thread.start()
                # best_move = find_best_move(self.board_bc, self.search_time, 999)
                # self.board.push(chess.Move.from_uci(bestWW_move.uci()))
                # self.board_bc.apply(best_move)
                # self.draw_board()
                # self.root.update()
                # self.root.after(30, self.engine_move)
        except Exception as e:
            print(f"Engine error: {e}")
    
    def _search_and_move(self):
        try:
            best_move = find_best_move(self.board_bc, self.search_time, 999)
            self.root.after(0, self._apply_engine_move, best_move)
        except Exception as e:
            print(f"SEARCH ERROR {e}")


    def _apply_engine_move(self, move: bc.Move):
        try:
            self.board_bc.apply(move)
            self.board.push(chess.Move.from_uci(move.uci()))
            self.draw_board()
            self.root.update()
            self.engine_move_pending = False
            self.root.after(30, self.engine_move)
        except Exception as e:
            print(f"ERROR APPLYING MOVE {e}")

    def can_move(self):
        """Check if human player can move"""
        white_is_cpu = self.white_player.get() == "cpu"
        black_is_cpu = self.black_player.get() == "cpu"
        
        if self.board.turn == chess.WHITE and white_is_cpu:
            return False
        if self.board.turn == chess.BLACK and black_is_cpu:
            return False
        return True
    
    def draw_board(self):
        self.canvas.delete("all")
        square_size = self.canvas_size // 8
        
        for row in range(8):
            for col in range(8):
                x1 = col * square_size
                y1 = row * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                color = "#F0D9B5" if (row + col) % 2 == 0 else "#B58863"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)
                
                square = chess.Square((7 - row) * 8 + col)
                piece = self.board.piece_at(square)
                if piece and square != self.selected_square:
                    piece_unicode = self.piece_to_unicode(piece.symbol())
                    text_color = "white" if piece.color else "black"
                    self.canvas.create_text(x1 + square_size // 2, y1 + square_size // 2,
                                          text=piece_unicode, font=("Arial", 50), fill=text_color)
    
    def piece_to_unicode(self, symbol):
        pieces = {'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
                 'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'}
        return pieces.get(symbol, symbol)
    
    def reset(self):
        """Reset the board"""
        self.board_bc = bc.Board()
        self.board = chess.Board(fen=self.board_bc.fen())
        self.selected_square = None
        self.draw_board()
        self.root.after(75, self.engine_move)
    
    def load_fen(self):
        """Load position from FEN string"""
        fen = simpledialog.askstring("Load FEN", "Enter FEN string:")
        if fen:
            try:
                self.board = chess.Board(fen)
                self.board_bc = bc.Board(fen)
                self.selected_square = None
                self.draw_board()
                self.root.after(150, self.engine_move)
            except:
                pass


if __name__ == "__main__":
    root = tk.Tk()
    gui = ChessGUI(root)
    root.mainloop()
