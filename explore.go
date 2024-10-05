package main

import (
	"flag"
	"fmt"
	"log"
	"sync"

	"github.com/notnil/chess"
)

const maxWorkers = 10

func exploreMoves(game *chess.Game, depth int, wg *sync.WaitGroup, pool chan struct{}) {
	defer wg.Done()
	fmt.Printf("ROUND %d", 40-depth)

	if depth == 0 {
		return
	}

	legalMoves := game.Position().ValidMoves()
	for _, move := range legalMoves {
		useGame := game.Clone()
		useGame.Move(move)

		wg.Add(1) //add to wait group

		pool <- struct{}{} //acquire worker

		go func(game *chess.Game, depth int) {
			defer func() {
				<-pool //Release the worker to the pool
				wg.Done()
			}()

			exploreMoves(useGame, depth-1, wg, pool)
		}(useGame, depth-1)
	}
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

	fmt.Printf("ROUND 1")
	err := game.Move(moves[moveIndex])
	if err != nil {
		log.Fatalf("Error making move: %v", err)
	}

	pool := make(chan struct{}, maxWorkers)

	var wg sync.WaitGroup
	wg.Add(1)

	go exploreMoves(game, 40, &wg, pool)

	wg.Wait()

}
