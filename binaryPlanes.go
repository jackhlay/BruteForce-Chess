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
	planes := make([][][]int, 12)
	for i := range planes {
		planes[i] = make([][]int, 8)
		for j := range planes[i] {
			planes[i][j] = make([]int, 8)
		}
	}
	//parse the fen string, and run the pieces into their respective planes

	parts := strings.Split(fen, " ")
	rows := []rune(parts[0])
	rank := 0
	file := 0

	for _, char := range rows {
		if char >= '1' && char <= '8' {
			file += int(char - '0')
		} else if char == '/' {
			rank++
			file = 0
		} else {
			planes[pieceCharToIndex(char)][rank][file] = 1
			file++
		}

		tensorData := make([]float64, 12*8*8)
		for i := 0; i < 12; i++ {
			for r := 0; r < 8; r++ {
				for f := 0; f < 8; f++ {
					tensorData[i*8*8+r*8+f] = float64(planes[i][r][f])
				}
			}
		}
		return tensor.New(tensor.WithShape(1, 12, 8, 8), tensor.WithBacking(tensorData)), nil
	}
	return nil, nil
}
