package feedforward

import (
	"math/rand"
	"sync"
	"time"
)

// Represents a multilayer feedforward neural network trained using the online variant of SGD.
type Network struct {
	BaseSubject
	neurons     []int
	activations []ActivationFunction
	layers      []layer
	initializer Initializer
	stop        StoppingCondition
	eta         float64
	isFitted    bool
}

// Constructor of a neural network.
func NewNetwork(neurons []int, activations []ActivationFunction, initializer Initializer, stop StoppingCondition, eta float64) *Network {
	return &Network{
		neurons:     neurons,
		activations: activations,
		layers:      constructLayers(neurons, activations),
		initializer: initializer,
		stop:        stop,
		eta:         eta,
	}
}

// Constructs all layers of given specification.
func constructLayers(neurons []int, activations []ActivationFunction) []layer {
	weights := constructWeights(neurons)
	biases := constructBiases(neurons)

	layerCount := len(neurons) - 1
	hiddenLayerCount := layerCount - 1

	layers := make([]layer, layerCount)
	for h := 0; h < hiddenLayerCount; h++ {
		layers[h] = newHiddenLayer(weights[h], biases[h], weights[h+1], activations[h])
	}
	layers[hiddenLayerCount] = newOutputLayer(weights[hiddenLayerCount], biases[hiddenLayerCount], activations[hiddenLayerCount])

	return layers
}

// Construct a 3d slice which holds all the weights used in the network.
func constructWeights(neurons []int) [][][]float64 {
	layerCount := len(neurons) - 1
	weights := make([][][]float64, layerCount)
	for k := 0; k < layerCount; k++ {
		weights[k] = make([][]float64, neurons[k])
		for i := 0; i < len(weights[k]); i++ {
			weights[k][i] = make([]float64, neurons[k+1])
		}
	}
	return weights
}

// Construct a 2d slice which holds all the biases used in the network.
func constructBiases(neurons []int) [][]float64 {
	layerCount := len(neurons) - 1
	biases := make([][]float64, layerCount)
	for k := 0; k < layerCount; k++ {
		biases[k] = make([]float64, neurons[k+1])
	}
	return biases
}

// Fits model to given sample using online SGD.
// Initializes weights on every call, doing so concurrently on a per layer basis.
func (n *Network) Fit(samples []Sample) {
	n.isFitted = true

	var wg sync.WaitGroup
	wg.Add(len(n.layers))
	for _, l := range n.layers {
		go func(layer layer) {
			defer wg.Done()
			layer.initialize(n.initializer)
		}(l)
	}
	wg.Wait()

	n.backpropagation(samples)
}

// Backpropagation main loop.
// Trains the network until the StoppingCondition is met and notifies ModelObserver instances currently subscribed to the network.
// Before starting an epoch a preprocess function is called which shuffles the samples.
func (n *Network) backpropagation(samples []Sample) {
	iter := 0
	for {
		statistics := NewIterationStatistic(iter, func() float64 { return MeanSquareError(n, samples) })

		n.NotifyObservers(statistics)

		if n.stop.IsMet(statistics) {
			break
		}

		preprocess(samples)
		n.completeEpoch(samples)
		iter++
	}
}

// Preprocess function which shuffles the samples before every epoch.
func preprocess(samples []Sample) {
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(samples), func(i, j int) { samples[i], samples[j] = samples[j], samples[i] })
}

// Performs an epoch of online SGD.
func (n *Network) completeEpoch(samples []Sample) {
	for _, sample := range samples {
		input := sample.Input
		expected := sample.Output
		actual := n.Predict(input)
		diff := make([]float64, len(expected))
		for i := 0; i < len(diff); i++ {
			diff[i] = expected[i] - actual[i]
		}

		for k := len(n.layers) - 1; k >= 0; k-- {
			delta := n.layers[k].processError(diff)
			var prevLayerOutput []float64
			if k != 0 {
				prevLayerOutput = n.layers[k-1].getOutputCache()
			} else {
				prevLayerOutput = input
			}

			layerWeight := n.layers[k].getWeights()
			layerBias := n.layers[k].getBiases()

			for i := 0; i < len(layerWeight); i++ {
				for j := 0; j < len(layerWeight[i]); j++ {
					layerWeight[i][j] += n.eta * delta[j] * prevLayerOutput[i]
				}
			}
			for i := 0; i < len(layerBias); i++ {
				layerBias[i] += n.eta * delta[i]
			}

			diff = delta
		}
	}
}

// Performs a model prediction.
func (n *Network) Predict(input []float64) []float64 {
	if !n.isFitted {
		panic("This instance of Network has not been fitted yet.")
	}

	output := n.layers[0].processInput(input)
	for i := 1; i < len(n.layers); i++ {
		output = n.layers[i].processInput(output)
	}
	return output
}
