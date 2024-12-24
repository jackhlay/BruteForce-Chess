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

var hostString = "stockfish.default.svc.cluster.local:4000"

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
func sfEval(fen string) float64 {
	// Connect to the Stockfish WebSocket server, passed in from GetConn()
	conn := GetConn()
	defer conn.Close()
	scoreChan := make(chan float64)
	errChan := make(chan error)

	go listenForMessages(conn, scoreChan, errChan)

	// Example to start the proces
	println(fen)
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
		if strings.Contains(payload, "depth 2") {
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
	sendCommand(conn, "isready")
	time.Sleep(100 * time.Millisecond) // Sleep briefly to let the engine initialize

	log.Println("Looking for evaluation score (depth=2)...")
	uciCommands := []string{
		"ucinewgame",
		fmt.Sprintf("position fen %s", fen),
		"go depth 2",
	}
	for _, command := range uciCommands {
		sendCommand(conn, command)
		time.Sleep(110 * time.Millisecond) // Sleep briefly to avoid busy waiting
	}
	return nil
}

// sendCommand sends a command to the Stockfish engine.
func sendCommand(conn *websocket.Conn, command string) {
	cmdMsg := Message{
		Type:    "uci:command",
		Payload: command,
	}
	log.Printf("Sending command: %s", command) // Add this line for debugging
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
