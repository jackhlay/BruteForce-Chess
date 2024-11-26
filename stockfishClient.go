package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/websocket"
)

var hostString = "localhost:4000" //Hits the container, and gets logged. Container ignores the input though...

var (
	isEngineReady   bool
	socketPath      = fmt.Sprint("ws://%s", hostString)
	evaluationScore string // This variable will hold the evaluation score
)

type Message struct {
	Type    string `json:"type"`
	Token   string `json:"token,omitempty"`
	Payload string `json:"payload,omitempty"`
}

// clientMain initializes the WebSocket connection and starts the process.
func SFmain() {
	u := url.URL{Scheme: "ws", Host: hostString, Path: "/"}
	fmt.Println("Connecting to", u.String())

	conn, _, err := websocket.DefaultDialer.Dial(u.String(), nil)
	if err != nil {
		log.Fatal("Dial error:", err)
	}
	time.Sleep(time.Millisecond * 700)
	defer conn.Close()

	go listenForMessages(conn)

	// Example to start the process
	fen := "3b2K1/6pp/3B4/5b2/3krP1n/q2p4/2p5/8 w - - 0 1" // Replace with your actual FEN string
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
		}
		if strings.Contains(payload, "depth 14") {
			re := regexp.MustCompile("score cp (-?\\d+)")

			match := re.FindStringSubmatch(payload)
			if match != nil {
				fmt.Printf("Matched Score: %s\n", match[1])
				scor, _ := strconv.Atoi((match[1]))
				evaluationScore = fmt.Sprintf("%.2f", float32(scor)/100)
			}

		}
	}
}

// processFen processes the FEN string and returns the evaluation score.
func processFen(conn *websocket.Conn, fen string) (string, error) {
	// Wait for the engine to be ready before sending any commands
	// if err := waitForEngineToBeReady(conn); err != nil {
	// 	return "", err
	// }

	sendCommand(conn, "ucinewgame")
	updateChessboard(fen)

	log.Println("Looking for evaluation score (depth=14)...")
	uciCommands := []string{
		fmt.Sprintf("position fen %s", fen),
		"go depth 8",
	}

	// Send commands to Stockfish
	for _, cmd := range uciCommands {
		log.Println(">> uci:command", cmd)
		sendCommand(conn, cmd)
	}

	// Wait for the evaluation score from Stockfish
	timeout := time.After(30 * time.Second)
	for evaluationScore == "" {
		select {
		case <-timeout:
			return "", fmt.Errorf("Timeout waiting for evaluation score")
		case <-time.After(100 * time.Millisecond): // Sleep briefly to avoid busy waiting
			continue
		}
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
	// Send the isready command to Stockfish
	sendCommand(conn, "isready")
	// Loop until we receive the "readyok" response
	timeout := time.After(30 * time.Second) // Timeout after 30 seconds
	for {
		select {
		case <-timeout:
			return fmt.Errorf("engine is not ready after waiting for 30 seconds")
		default:
			// Continuously check for messages
			_, message, err := conn.ReadMessage()
			if err != nil {
				log.Println("Read error:", err)
				return err
			}

			var eventData map[string]interface{}
			if err := json.Unmarshal(message, &eventData); err != nil {
				log.Println("Unmarshal error:", err)
				continue
			}

			// Check for the readyok message
			if eventData["type"] == "uci:response" {
				payload := eventData["payload"].(string)
				if payload == "readyok" {
					isEngineReady = true
					log.Println("Engine is ready")
					return nil
				}
			}
		}
	}
}

// updateChessboard simulates updating the chessboard with the FEN string.
func updateChessboard(fen string) {
	// Update your chessboard logic here
	log.Printf("Chessboard updated with FEN: %s", fen)
}
