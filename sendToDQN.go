package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

func sendJSON(data PosData) {
	//send to dqn
	jsonData, _ := json.Marshal(data)
	resp, err := http.Post("chessdqn.default.svc.cluster.local:8000", "application/json", bytes.NewBuffer(jsonData))
	// resp, err := http.Post("chessdqn.default.svc.cluster.local:8000", "application/json", bytes.NewBuffer(jsonData))

	if err != nil {
		fmt.Println("Error Making POST req: ", err)
	}

	defer resp.Body.Close()

	// Check the response
	if resp.StatusCode == http.StatusOK {
		fmt.Printf("200OK %s\n", resp.Body)
	} else if resp.StatusCode == http.StatusResetContent {
		fmt.Printf("205 - End of the line! %s\n", resp.Body)
	} else {
		fmt.Printf("Failed with status code: %d\n", resp.StatusCode)
	}
}
