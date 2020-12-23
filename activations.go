package feedforward

import "math"

type ActivationFunction interface {
	Value(net float64) float64
	Gradient(net float64) float64
}

func Repeat(activation ActivationFunction, n int) {
	activations := make([]ActivationFunction, n)
	for i := 0; i < n; i++ {
		activations[i] = activation
	}
}

type sigmoid struct{}

func Sigmoid() ActivationFunction {
	return sigmoid{}
}

func (s sigmoid) Value(net float64) float64 {
	return 1 / (1 + math.Exp(-net))
}

func (s sigmoid) Gradient(net float64) float64 {
	return net * (1 - net)
}
