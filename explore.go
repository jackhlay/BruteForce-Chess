package main

import (
	"flag"
	"fmt"
	"log"
	"sync"

	"github.com/notnil/chess"
)

const maxWorkers = 10

func exploreMoves(game *chess.Game, depth int, wg *sync.WaitGroup, pool chan struct{}, done chan struct{}) {
	defer wg.Done()
	fmt.Printf("ROUND %d\n", 17-depth)

	if depth == 0 {
		return
	}

	legalMoves := game.Position().ValidMoves()
	var wgDepth sync.WaitGroup // Wait group for the current depth level

	for _, move := range legalMoves {
		useGame := game.Clone()
		err := useGame.Move(move)
		if err != nil {
			log.Fatalf("Error making move: %v", err)
		}

		pool <- struct{}{} // Acquire worker
		wgDepth.Add(1)     // Add to depth wait group

		go func(game *chess.Game, depth int) {
			defer func() {
				<-pool // Release  the worker to the pool
				wgDepth.Done()
			}()
			fen := game.Position().String()
			//send the fen to stockfish pods and in house eval, then to dqn for training
			exploreMoves(game, depth-1, wg, pool, done)
		}(useGame, depth-1)
	}

	// Wait for all moves at the current depth level to complete
	go func() {
		wgDepth.Wait()     // Wait for all moves to finish
		done <- struct{}{} // Signal that we're done with this depth level
	}()
}

func main() {
	var moveIndex int
	flag.IntVar(&moveIndex, "move", 0, "Index of the opening move / pod number")
	flag.Parse()

	if moveIndex < 0 || moveIndex > 19 {
		log.Fatalf("INVALID INDEX: %d", moveIndex)
	}

	game := chess.NewGame()
	position := game.Position()
	moves := position.ValidMoves()

	fmt.Printf("ROUND 1\n")
	err := game.Move(moves[moveIndex])
	if err != nil {
		log.Fatalf("Error making move: %v", err)
	}

	pool := make(chan struct{}, maxWorkers)
	done := make(chan struct{}) // Channel to signal when depth is done

	var wg sync.WaitGroup
	wg.Add(1)

	go exploreMoves(game, 17, &wg, pool, done)

	// Wait for each depth level to finish
	for depth := 17; depth > 0; depth-- {
		<-done
		fmt.Printf("Finished ROUND %d\n", 17-depth)
	}

	wg.Wait()
}
