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

var hostString = "stockfish:4000"

type EngineConf struct {
	isEngineReady   bool
	socketPath      string
	evaluationScore float64
}

var engineConf = EngineConf{
	isEngineReady:   false,
	socketPath:      fmt.Sprintf("ws://%s", hostString),
	evaluationScore: 0.0,
}

type Message struct {
	Type    string `json:"type"`
	Token   string `json:"token,omitempty"`
	Payload string `json:"payload,omitempty"`
}

func GetConn() *websocket.Conn {
	u := url.URL{Scheme: "ws", Host: hostString, Path: "/"}

	conn, _, err := websocket.DefaultDialer.Dial(u.String(), nil)
	if err != nil {
		log.Fatal("Dial error:", err)
	}
	time.Sleep(time.Millisecond * 700)
	return conn
}

// clientMain initializes the WebSocket connection and starts the process.
func sfEval(conn *websocket.Conn, fen string) float64 {
	// Connect to the Stockfish WebSocket server, passed in from GetConn()

	scoreChan := make(chan float64)
	errChan := make(chan error)

	go listenForMessages(conn, scoreChan, errChan)

	// Example to start the proces
	println(fen)
	// fens := "3b2K1/6pp/3B4/5b2/3krP1n/q2p4/2p5/8 w - - 0 1" // Replace with your actual FEN string
	if err := processFen(conn, fen, scoreChan, errChan); err != nil {
		log.Fatal("Error processing FEN:", err)
	}

	select {
	case score := <-scoreChan:
		fmt.Println("Evaluation score:", score)
		return score
	case err := <-errChan:
		log.Println("Error:", err)
		return -9999.9
	}
}

// listenForMessages continuously listens for messages from Stockfish.
func listenForMessages(conn *websocket.Conn, scoreChan chan float64, errChan chan error) {
	for {
		_, message, err := conn.ReadMessage()
		if err != nil {
			log.Println("Read error:", err)
			return
		}
		score, err := handleMessage(message)
		if err != nil {
			errChan <- err
		} else if score != nil {
			scoreChan <- *score
		}
	}
}

// handleMessage extracts the evaluation score from incoming messages.
func handleMessage(message []byte) (*float64, error) {
	var eventData map[string]interface{}
	if err := json.Unmarshal(message, &eventData); err != nil {
		return nil, fmt.Errorf("unmarshal error: %w", err)
	}

	fmt.Printf("Received message: %s\n", message)

	if eventData["type"] == "uci:response" {
		payload := eventData["payload"].(string)
		log.Println("<< uci:response", payload)

		// Check for "readyok"
		if payload == "readyok" {
			engineConf.isEngineReady = true
			return nil, nil
		}

		// Extract score
		if strings.Contains(payload, "depth 8") {
			re := regexp.MustCompile(`score cp (-?\d+)`)
			match := re.FindStringSubmatch(payload)
			if match != nil {
				score, err := strconv.Atoi(match[1])
				if err != nil {
					return nil, fmt.Errorf("invalid score format: %w", err)
				}
				scoreFloat := float64(score)
				return &scoreFloat, nil
			}
		}
	}
	return nil, nil
}

// processFen processes the FEN string and returns the evaluation score.
func processFen(conn *websocket.Conn, fen string, scoreChan chan float64, errChan chan error) error {
	// Wait for the engine to be ready before sending any commands
	// if err := waitForEngineToBeReady(conn); err != nil {
	// 	return -1.0, err
	// }

	sendCommand(conn, "ucinewgame")

	log.Println("Looking for evaluation score (depth=8)...")
	uciCommands := []string{
		fmt.Sprintf("ucinewgame", "position fen %s", fen),
		"go depth 8",
	}

	// Send commands to Stockfish
	for _, cmd := range uciCommands {
		log.Println(">> uci:command", cmd)
		sendCommand(conn, cmd)
		time.Sleep(17 * time.Millisecond) // Sleep briefly to avoid busy waiting
	}
	return nil
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

			if strings.Contains(string(message), "readyok") {
				engineConf.isEngineReady = true
				return nil
			}
		}
	}
}
