package feedforward

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

// Type which models a standard floating point sample,
// consisting of any number of floating point inputs and any number of floating point outputs
type Sample struct {
	Input  []float64
	Output []float64
}

type Delimiters struct {
	InputValues  string
	InputOutput  string
	OutputValues string
}

// Function for loading samples from a text file into a slice
func Load(path string, delimiters Delimiters) ([]Sample, error) {
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

		rawSample := strings.Split(scanner.Text(), delimiters.InputOutput)
		rawInput := strings.Split(rawSample[0], delimiters.InputValues)
		rawOutput := strings.Split(rawSample[1], delimiters.OutputValues)

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
