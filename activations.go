package feedforward

import "math"

type ActivationFunction interface {
	Value(net float64) float64
	Gradient(net float64) float64
}

type Sigmoid struct{}

func (s Sigmoid) Value(net float64) float64 {
	return 1 / (1 + math.Exp(-net))
}

func (s Sigmoid) Gradient(net float64) float64 {
	return net * (1 - net)
}
