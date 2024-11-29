package main

import (
	"flag"
	"fmt"
	"log"
	"sync"

	"github.com/notnil/chess"
)

type PosData struct {
	StartFen    string  `json:"start_fen"`
	StartRating float64 `json:"start_rating"`
	Action      string  `json:"action"`
	EndFen      string  `json:"end_fen"`
	EndRating   float64 `json:"end_rating"`
}

const maxWorkers = 4

func exploreMoves(game *chess.Game, depth int, wg *sync.WaitGroup, pool chan struct{}) {
	defer wg.Done()
	fmt.Printf("ROUND %d\n", 17-depth)

	if depth == 0 {
		return
	}

	legalMoves := game.Position().ValidMoves()

	for _, move := range legalMoves {
		useGame := game.Clone()
		startfen := useGame.Position().String()
		err := useGame.Move(move)
		if err != nil {
			log.Fatalf("Error making move: %v", err)
		}
		endfen := useGame.Position().String()
		endRating := sfEval(endfen)
		startRating := sfEval(startfen)

		pool <- struct{}{} // Acquire worker slot
		wg.Add(1)

		go func(game *chess.Game, depth int, startfen, endfen string) {
			defer func() {
				<-pool // Release the worker to the pool
				wg.Done()
			}()

			data := PosData{
				StartFen:    startfen,
				StartRating: startRating,
				Action:      move.String(),
				EndFen:      endfen,
				EndRating:   endRating,
			}
			fmt.Println(data)
			sendJSON(data)
			// Send data to DQN or handle further as needed

			// Recurse into next depth
			exploreMoves(game, depth-1, wg, pool)
		}(useGame, depth-1, startfen, endfen) // Pass startfen and endfen to the goroutine
	}
}

func main() {
	var moveIndex int
	flag.IntVar(&moveIndex, "move", 0, "Index of the opening move / pod number")
	flag.Parse()

	if moveIndex < 0 || moveIndex > 19 {
		// log.Fatalf("INVALID INDEX: %d", moveIndex)
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

	// rating := sfEval(endfen)

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

	go exploreMoves(game, 15, &wg, pool)

	wg.Wait()
	fmt.Println("ALL MOVES EXPLORED")
}
