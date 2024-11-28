package main

import (
	"flag"
	"fmt"
	"log"
	"net/url"
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
		startfen := useGame.Position().String()
		err := useGame.Move(move)
		if err != nil {
			log.Fatalf("Error making move: %v", err)
		}
		endfen := useGame.Position().String()

		pool <- struct{}{} // Acquire worker
		wgDepth.Add(1)     // Add to depth wait group

		go func(game *chess.Game, depth int, startfen, endfen string) {
			defer func() {
				<-pool // Release the worker to the pool
				wgDepth.Done()
			}()

			//Stockfish connection / interaction
			socketPath := "ws://localhost:4000"
			u, err := url.Parse(socketPath)
			if err != nil {
				log.Fatalf("Error parsing WebSocket URL: %v", err) // Fixed error handling
			}

			conn, _, err := websocket.DefaultDialer.Dial(u.String(), nil)
			if err != nil { // Added error handling for WebSocket connection
				log.Fatalf("Error connecting to WebSocket: %v", err)
			}

			startrating, err := processFen(conn, startfen) // Added error handling
			if err != nil {
				log.Fatalf("Error processing start FEN: %v", err)
			}

			endrating, err := processFen(conn, endfen) // Added error handling
			if err != nil {
				log.Fatalf("Error processing end FEN: %v", err)
			}
			conn.Close()

			data := PosData{
				StartFen:    startfen,
				StartRating: startrating,
				Action:      move.String(),
				EndFen:      endfen,
				EndRating:   endrating,
			}

			sendJSON(data)
			// Send data to DQN or handle further as needed
			// Update pod name in redis with fen string and depth in case of crash
			// Here you would send `data` to your DQN for training

			// Continue exploring moves recursively
			exploreMoves(game, depth-1, wg, pool, done)
		}(useGame, depth-1, startfen, endfen) // Pass startfen and endfen to the goroutine
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

	socketPath := "ws://10.0.0.112:4000"
	u, _ := url.Parse(socketPath)
	conn, _, _ := websocket.DefaultDialer.Dial(u.String(), nil)
	rating, _ := processFen(conn, endfen)
	conn.Close()

	data := PosData{
		StartFen:    startfen,
		StartRating: 1500,
		Action:      moves[moveIndex].String(),
		EndFen:      endfen,
		EndRating:   float64(rating),
	}
	sendJSON(data)

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
