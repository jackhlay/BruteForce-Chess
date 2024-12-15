package main

import (
	"flag"
	"fmt"
	"log"
	"sync"

	"github.com/gorilla/websocket"
	"github.com/notnil/chess"
)

type PosData struct {
	StartFen    string  `json:"start_fen"`
	StartRating float64 `json:"start_rating"`
	Action      string  `json:"action"`
	EndFen      string  `json:"end_fen"`
	EndRating   float64 `json:"end_rating"`
}

const maxWorkers = 8

func exploreMoves(pos *chess.Position, depth int, wg *sync.WaitGroup, pool chan struct{}, conn *websocket.Conn) {
	defer wg.Done()

	if depth == 0 {
		return
	}

	legalMoves := pos.ValidMoves()
	for _, move := range legalMoves {
		pool <- struct{}{} // Acquire worker slot
		wg.Add(1)

		go func(move *chess.Move, Pos *chess.Position) {
			defer func() {
				<-pool // Release the worker slot
				wg.Done()
				if r := recover(); r != nil {
					log.Printf("Recovered from panic: %v", r)
				}
			}()

			// Clone the position to avoid data races
			newPos := Pos.Update(move)

			startfen := Pos.String()
			endfen := newPos.String()
			startRating := sfEval(startfen)
			endRating := sfEval(endfen)

			data := PosData{
				StartFen:    startfen,
				StartRating: startRating,
				Action:      move.String(),
				EndFen:      endfen,
				EndRating:   endRating,
			}
			fmt.Println(data)
			sendJSON(data)

			// Recurse into the next depth
			exploreMoves(newPos, depth-1, wg, pool, conn)
		}(move, pos)
	}
}

func main() {
	conn := GetConn()
	defer conn.Close()

	var moveIndex int
	flag.IntVar(&moveIndex, "move", 0, "Index of the opening move / pod number")
	flag.Parse()

	if moveIndex < 0 || moveIndex > 19 {
		moveIndex = 0
	}

	game := chess.NewGame()
	position := game.Position()
	moves := position.ValidMoves()
	startfen := position.String()

	fmt.Printf("ROUND 1\n")
	err := game.Move(moves[moveIndex])
	if err != nil {
		log.Fatalf("Error making move: %v", err)
	}

	endfen := position.String()
	endRating := sfEval(endfen)

	data := PosData{
		StartFen:    startfen,
		StartRating: 0,
		Action:      moves[moveIndex].String(),
		EndFen:      endfen,
		EndRating:   endRating,
	}
	sendJSON(data)

	pool := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup

	wg.Add(1)
	pool <- struct{}{}

	go exploreMoves(game.Position(), 15, &wg, pool, conn)

	wg.Wait()
	fmt.Println("ALL MOVES EXPLORED")
}
