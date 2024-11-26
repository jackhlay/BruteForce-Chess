package main

import (
	"strings"

	"gorgonia.org/tensor"
)

// Convert a piece character to a binary plane index
func pieceCharToIndex(piece rune) int {
	switch piece {
	case 'P':
		return 0 // White Pawn
	case 'N':
		return 1 // White Knight
	case 'B':
		return 2 // White Bishop
	case 'R':
		return 3 // White Rook
	case 'Q':
		return 4 // White Queen
	case 'K':
		return 5 // White King
	case 'p':
		return 6 // Black Pawn
	case 'n':
		return 7 // Black Knight
	case 'b':
		return 8 // Black Bishop
	case 'r':
		return 9 // Black Rook
	case 'q':
		return 10 // Black Queen
	case 'k':
		return 11 // Black King
	default:
		return -1 // No piece
	}
}

func makePlanes(fen string) (*tensor.Dense, error) {
	// Initialize the planes (12 planes for the 12 piece types on an 8x8 board)
	planes := make([]float32, 12*boardSize*boardSize)

	// Split FEN string into its components
	parts := strings.Split(fen, " ")
	rows := []rune(parts[0]) // The board layout is the first part of the FEN
	rank := 0
	file := 0

	// Iterate over the characters in the FEN string (the board layout)
	for _, char := range rows {
		if char >= '1' && char <= '8' {
			// A number means empty squares. Skip the number of squares.
			file += int(char - '0')
		} else if char == '/' {
			// New rank (row) in the FEN string
			rank++
			file = 0
		} else {
			// For pieces, determine the correct plane index and set the square to 1
			pieceIndex := pieceCharToIndex(char)
			planes[pieceIndex*boardSize*boardSize+rank*boardSize+file] = 1
			file++
		}
	}

	// Create a tensor with shape (1, 12, 8, 8) to represent the planes
	return tensor.New(tensor.WithShape(1, 12, boardSize, boardSize), tensor.WithBacking(planes)), nil
}
