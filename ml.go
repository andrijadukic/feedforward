package feedforward

// Defines a machine learning model.
// Every type that implements this interface has a Fit phase and a Predict phase.
// Fit phase must be called before calling Predict the first time.
// Implementations should signal if Predict is called before Fit.
type Model interface {
	Fit([]Sample)
	Predict([]float64) []float64
}

// Defines a machine learning model which can be trained without seeing all the samples.
type IterativeModel interface {
	Model
	PartialFit([]Sample)
}
