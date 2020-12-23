package feedforward

import "math"

// Represents a neural activation function.
type ActivationFunction struct {
	Value    func(net float64) float64
	Gradient func(net float64) float64
}

// Helper function to fill a slice of size n with the given ActivationFunction.
func Repeat(activation ActivationFunction, n int) {
	activations := make([]ActivationFunction, n)
	for i := 0; i < n; i++ {
		activations[i] = activation
	}
}

// Sigmoid activation function.
func Sigmoid() ActivationFunction {
	return ActivationFunction{
		Value:    func(net float64) float64 { return 1 / (1 + math.Exp(-net)) },
		Gradient: func(net float64) float64 { return net * (1 - net) },
	}
}

// TanH activation function.
func TanH() ActivationFunction {
	return ActivationFunction{
		Value:    func(net float64) float64 { return (1 - math.Exp(-2*net)) / (1 + math.Exp(-2*net)) },
		Gradient: func(net float64) float64 { return 1 - math.Pow(net, 2) },
	}
}

// ReLu activation function.
func ReLu() ActivationFunction {
	return ActivationFunction{
		Value: func(net float64) float64 { return math.Max(net, 0) },
		Gradient: func(net float64) float64 {
			if net > 0 {
				return 1
			}
			return 0
		},
	}
}
