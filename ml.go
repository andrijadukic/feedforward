package feedforward

import "math"

// Defines a machine learning model.
// Inspired by sklearn.
// Every type that implements this interface has a Fit phase and a Predict phase.
// Fit phase must be called before calling Predict the first time.
// Implementations should throw an error if Predict is called before Fit.
type Model interface {
	Fit([]Sample)
	Predict([]float64) []float64
}

// Defines a machine learning model which can be trained without seeing all the samples.
type IterativeModel interface {
	Model
	Predict([]float64) []float64
}

// MSE loss function.
func MeanSquareError(model Model, samples []Sample) float64 {
	mse := 0.
	for _, sample := range samples {
		expected := sample.Output
		actual := model.Predict(sample.Input)
		for i := 0; i < len(actual); i++ {
			mse += math.Pow(expected[i]-actual[i], 2)
		}
	}
	return mse / float64(len(samples))
}
