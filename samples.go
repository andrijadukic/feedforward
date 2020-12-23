package feedforward

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

type Sample struct {
	Input  []float64
	Output []float64
}

func Load(path string) ([]Sample, error) {
	var samples []Sample

	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		if err := scanner.Err(); err != nil {
			return nil, err
		}

		rawSample := strings.Split(scanner.Text(), " -> ")
		rawInput := strings.Split(rawSample[0], ",")
		rawOutput := strings.Split(rawSample[1], ",")

		input := make([]float64, len(rawInput))
		output := make([]float64, len(rawOutput))

		for i := 0; i < len(input); i++ {
			val, err := strconv.ParseFloat(rawInput[i], 64)
			if err != nil {
				return nil, err
			}
			input[i] = val
		}

		for i := 0; i < len(output); i++ {
			val, err := strconv.ParseFloat(rawOutput[i], 64)
			if err != nil {
				return nil, err
			}
			output[i] = val
		}

		samples = append(samples, Sample{Input: input, Output: output})
	}
	if err := file.Close(); err != nil {
		return nil, err
	}

	return samples, nil
}
