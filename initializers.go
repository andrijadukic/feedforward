package feedforward

import (
	"math/rand"
	"time"
)

// Represents a weights initializer
type Initializer interface {
	Initialize([][]float64)
}

// Type to hold the interval in which the weights are to be initialized.
type uniform struct {
	lb float64
	ub float64
}

// Constructor of a uniform initializer.
func NewUniformInitializer(lb, ub float64) Initializer {
	return &uniform{
		lb: lb,
		ub: ub,
	}
}

// Uniform initialization function.
func (u *uniform) Initialize(weights [][]float64) {
	rand.Seed(time.Now().UTC().UnixNano())
	for i := 0; i < len(weights); i++ {
		for j := 0; j < len(weights[i]); j++ {
			weights[i][j] = u.lb + rand.Float64()*(u.ub-u.lb)
		}
	}
}
