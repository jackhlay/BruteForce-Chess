package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"sync"
	"time"

	"github.com/notnil/chess"
)

// type PosData struct {
// 	StartFen    string  `json:"start_fen"`
// 	StartRating float64 `json:"start_rating"`
// 	Action      string  `json:"action"`
// 	EndFen      string  `json:"end_fen"`
// 	EndRating   float64 `json:"end_rating"`
// }

type Work struct {
	pos    *chess.Position
	move   string
	depth  int
	parent *Work
}

const maxWorkers = 2

func worker(workQueue chan Work, pool chan struct{}, wg *sync.WaitGroup) {
	for work := range workQueue {
		endRating := 0.0
		pool <- struct{}{} // Acquire worker slot
		wg.Add(1)
		go func(work Work) {
			defer func() {
				if r := recover(); r != nil {
					log.Printf("Recovered from panic: %v", r)
				}
			}()
			start := work.pos.String()
			fen, _ := chess.FEN(start)
			startEval := sfEval(start)
			game := chess.NewGame(fen)
			game.MoveStr(work.move)
			endfen := game.Position().String()
			if game.Outcome() == chess.NoOutcome {
				endRating = sfEval(endfen)
			} else if game.Outcome() == chess.Draw {
				endRating = 0
				work.depth = 0
			} else {
				if game.Outcome() == chess.WhiteWon {
					endRating = math.Inf(1)
				}
				if game.Outcome() == chess.BlackWon {
					endRating = math.Inf(-1)
				}
				work.depth = 0
			}

			data := PosData{
				StartFen:    start,
				StartRating: startEval / 100,
				Action:      work.move,
				EndFen:      endfen,
				EndRating:   endRating / 100,
			}
			fmt.Println(data)
			sendJSON(data)

			<-pool // Release the worker slot
			wg.Done()

			// Recurse into the next depth
			time.Sleep(150 * time.Millisecond)
			exploreMoves(game.Position(), work.depth, wg, workQueue)
		}(work)
	}
}

func exploreMoves(pos *chess.Position, depth int, wg *sync.WaitGroup, workQueue chan Work) {
	defer wg.Done()

	if depth == 0 {
		return
	}

	legalMoves := pos.ValidMoves()
	for _, move := range legalMoves {
		fen, _ := chess.FEN(pos.String())
		newGame := chess.NewGame(fen)
		newPos := newGame.Position()

		moveStr := move.String()

		// Enqueue new work
		wg.Add(1)
		workQueue <- Work{newPos, moveStr, depth - 1, nil}
	}
}

func main() {
	workQueue := make(chan Work, 1024)

	var moveIndex int
	flag.IntVar(&moveIndex, "move", 0, "Index of the opening move / pod number")
	flag.Parse()

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
		EndRating:   endRating / 100,
	}
	sendJSON(data)

	pool := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup

	for i := 0; i < maxWorkers; i++ {
		go worker(workQueue, pool, &wg)
	}

	wg.Add(1)
	workQueue <- Work{game.Position(), "", 7, nil}

	wg.Wait()
	go exploreMoves(game.Position(), 7, &wg, workQueue)

	wg.Wait()
	close(workQueue)
	fmt.Println("ALL MOVES EXPLORED")
}
