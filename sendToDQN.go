package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

func sendJSON(url string, data PosData) {
	//send to dqn
	jsonData, _ := json.Marshal(data)
	resp, err := http.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		fmt.Println("Error Making POST req: ", err)
	}

	defer resp.Body.Close()

	// Check the response
	if resp.StatusCode == http.StatusOK {
		fmt.Println(resp)
	} else {
		fmt.Printf("Failed with status code: %d\n", resp.StatusCode)
	}
}
