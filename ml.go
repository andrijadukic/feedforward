package feedforward

import "math"

type Model interface {
	Fit([]Sample)
	Predict([]float64) []float64
}

type IterativeModel interface {
	Model
	Predict([]float64) []float64
}

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
