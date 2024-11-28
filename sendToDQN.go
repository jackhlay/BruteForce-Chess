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
	resp, err := http.Post("http://127.0.0.1:8000/", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		fmt.Println("Error Making POST req: ", err)
	}

	defer resp.Body.Close()

	// Check the response
	if resp.StatusCode == http.StatusOK {
		fmt.Printf("200OK %s\n", resp.Body)
	} else {
		fmt.Printf("Failed with status code: %d\n", resp.StatusCode)
	}
}