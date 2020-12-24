package feedforward

import (
	"math"
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

// Type to hold the parameters of normal distribution to be used for weight initialization
type gaussian struct {
	mean   float64
	stddev float64
}

// Constructor of a gaussian initializer.
func NewGaussianInitializer(mean, stddev float64) Initializer {
	return &gaussian{
		mean:   mean,
		stddev: stddev,
	}
}

// Gaussian initialization function.
func (g *gaussian) Initialize(weights [][]float64) {
	rand.Seed(time.Now().UTC().UnixNano())
	for i := 0; i < len(weights); i++ {
		for j := 0; j < len(weights[i]); j++ {
			weights[i][j] = g.stddev*rand.NormFloat64() + g.mean
		}
	}
}

// Type to hold the parameters of normal distribution to be used for weight initialization
type xavier struct{}

// Constructor of a Xavier initializer.
func NewXavierInitializer(mean, stddev float64) Initializer {
	return xavier{}
}

// Xavier initialization function.
func (x xavier) Initialize(weights [][]float64) {
	rand.Seed(time.Now().UTC().UnixNano())
	n := len(weights)
	for i := 0; i < n; i++ {
		m := len(weights[i])
		bound := math.Sqrt(float64(6. / (n + m)))
		for j := 0; j < m; j++ {
			weights[i][j] = -bound + rand.Float64()*2*bound
		}
	}
}
