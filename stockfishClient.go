package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/url"
	"time"

	"github.com/gorilla/websocket"
)

var (
	isEngineReady   bool
	socketPath      = "ws://localhost:4000"
	evaluationScore string // This variable will hold the evaluation score
)

type Message struct {
	Type    string `json:"type"`
	Token   string `json:"token,omitempty"`
	Payload string `json:"payload,omitempty"`
}

// clientMain initializes the WebSocket connection and starts the process.
func clientMain() {
	u := url.URL{Scheme: "ws", Host: "localhost:4000", Path: "/"}
	fmt.Println("Connecting to", u.String())

	conn, _, err := websocket.DefaultDialer.Dial(u.String(), nil)
	if err != nil {
		log.Fatal("Dial error:", err)
	}
	defer conn.Close()

	go listenForMessages(conn)

	// Example to start the process
	fen := "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" // Replace with your actual FEN string
	evaluation, err := processFen(conn, fen)
	if err != nil {
		log.Fatal("Error processing FEN:", err)
	}

	// Print the evaluation score
	fmt.Println("Evaluation Score:", evaluation)
}

// listenForMessages continuously listens for messages from Stockfish.
func listenForMessages(conn *websocket.Conn) {
	for {
		_, message, err := conn.ReadMessage()
		if err != nil {
			log.Println("Read error:", err)
			return
		}
		handleMessage(message)
	}
}

// handleMessage processes incoming messages from the Stockfish engine.
func handleMessage(message []byte) {
	var eventData map[string]interface{}
	if err := json.Unmarshal(message, &eventData); err != nil {
		log.Println("Unmarshal error:", err)
		return
	}

	fmt.Printf("Received message: %s\n", message)

	switch eventData["type"] {
	case "uci:response":
		payload := eventData["payload"].(string)
		log.Println("<< uci:response", payload)
		if payload == "readyok" {
			isEngineReady = true
		} else {
			evaluationScore = payload // Capture the evaluation score
		}
	}
}

// processFen processes the FEN string and returns the evaluation score.
func processFen(conn *websocket.Conn, fen string) (string, error) {
	updateChessboard(fen)

	log.Println(">> uci:command", "ucinewgame")
	sendCommand(conn, "ucinewgame")

	if err := waitForEngineToBeReady(conn); err != nil {
		return "", err // Return error if the engine isn't ready
	}

	log.Println("Looking for evaluation score (depth=25)...")
	uciCommands := []string{
		fmt.Sprintf("position fen %s", fen),
		"go depth 25",
	}

	// Send commands to Stockfish
	for _, cmd := range uciCommands {
		log.Println(">> uci:command", cmd)
		sendCommand(conn, cmd)
	}

	// Wait for the evaluation score from Stockfish
	for evaluationScore == "" { // Loop until we get a score
		time.Sleep(100 * time.Millisecond) // Sleep briefly to avoid busy waiting
	}

	return evaluationScore, nil
}

// sendCommand sends a command to the Stockfish engine.
func sendCommand(conn *websocket.Conn, command string) {
	cmdMsg := Message{
		Type:    "uci:command",
		Payload: command,
	}
	if err := conn.WriteJSON(cmdMsg); err != nil {
		log.Println("WriteJSON error:", err)
	}
}

// waitForEngineToBeReady waits until the engine is ready to receive commands.
func waitForEngineToBeReady(conn *websocket.Conn) error {
	isEngineReady = false
	elapsedTime := 0
	timeoutInMs := 30000

	sendCommand(conn, "isready")

	for !isEngineReady {
		if elapsedTime >= timeoutInMs {
			return fmt.Errorf("engine is not ready after waiting for 30 seconds")
		}
		time.Sleep(1 * time.Second)
		elapsedTime += 1000
	}
	return nil
}

// updateChessboard simulates updating the chessboard with the FEN string.
func updateChessboard(fen string) {
	// Update your chessboard logic here
	log.Printf("Chessboard updated with FEN: %s", fen)
}
