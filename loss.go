package feedforward

import (
	"math"
)

// Represents a function which takes a slice of float64 as input and returns a prediction of an output.
// This type is a generalization of the Predict method of Model interface as this does not require the Fit method, which
// is not relevant in model scoring.
type Predictor func([]float64) []float64

// Represents a loss function which takes a predictor function and the sample for the predictor function to be tested against
type LossFunction func(Predictor, []Sample) float64

// MSE loss function.
func MeanSquareError(predictor Predictor, samples []Sample) float64 {
	mse := 0.
	for _, sample := range samples {
		expected := sample.Output
		actual := predictor(sample.Input)
		for i := 0; i < len(actual); i++ {
			mse += math.Pow(expected[i]-actual[i], 2)
		}
	}
	return mse / float64(len(samples))
}
