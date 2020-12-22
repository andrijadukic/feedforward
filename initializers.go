package feedforward

import (
	"math/rand"
	"time"
)

type Initializer interface {
	Initialize([][]float64)
}

type uniform struct {
	lb float64
	ub float64
}

func NewUniformInitializer(lb, ub float64) Initializer {
	return &uniform{
		lb: lb,
		ub: ub,
	}
}

func (init *uniform) Initialize(weights [][]float64) {
	rand.Seed(time.Now().UTC().UnixNano())
	for i := 0; i < len(weights); i++ {
		for j := 0; j < len(weights[i]); j++ {
			weights[i][j] = init.lb + rand.Float64()*(init.ub-init.lb)
		}
	}
}
