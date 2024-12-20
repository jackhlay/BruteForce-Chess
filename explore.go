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

type Work struct {
	pos    *chess.Position
	move   string
	depth  int
	parent *Work
}

const maxWorkers = 8

func worker(workQueue chan Work, pool chan struct{}, wg *sync.WaitGroup) {
	for work := range workQueue {
		pool <- struct{}{} // Acquire worker slot
		wg.Add(1)
		go func(work Work) {
			defer func() {
				if r := recover(); r != nil {
					log.Printf("Recovered from panic: %v", r)
				}
			}()

			startfen := work.pos.String()
			endfen := work.pos.String()
			startRating := sfEval(startfen)
			endRating := sfEval(endfen)

			data := PosData{
				StartFen:    startfen,
				StartRating: startRating,
				Action:      work.move, // Update with actual move
				EndFen:      endfen,
				EndRating:   endRating,
			}
			fmt.Println(data)
			sendJSON(data)

			<-pool // Release the worker slot
			wg.Done()

			// Recurse into the next depth
			exploreMoves(work.pos, work.depth, wg, workQueue)
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
		newPos := pos.Update(move)
		wg.Add(1)
		moveStr := move.String()
		workQueue <- Work{newPos, moveStr, depth - 1, nil}
	}
}

func main() {
	workQueue := make(chan Work, 1024)

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

	for i := 0; i < maxWorkers; i++ {
		go worker(workQueue, pool, &wg)
	}

	wg.Add(1)
	workQueue <- Work{game.Position(), "", 15, nil}

	wg.Wait()
	go exploreMoves(game.Position(), 15, &wg, workQueue)

	wg.Wait()
	close(workQueue)
	fmt.Println("ALL MOVES EXPLORED")
}
