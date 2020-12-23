package feedforward

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

type Network struct {
	neurons     []int
	activations []ActivationFunction
	layers      []layer
	initializer Initializer
	eta         float64
}

func NewNetwork(neurons []int, activations []ActivationFunction, initializer Initializer, eta float64) Model {
	return &Network{
		neurons:     neurons,
		activations: activations,
		layers:      constructLayers(neurons, activations),
		initializer: initializer,
		eta:         eta,
	}
}

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

func constructBiases(neurons []int) [][]float64 {
	layerCount := len(neurons) - 1
	biases := make([][]float64, layerCount)
	for k := 0; k < layerCount; k++ {
		biases[k] = make([]float64, neurons[k+1])
	}
	return biases
}

func (n *Network) Fit(samples []Sample) {
	var wg sync.WaitGroup
	wg.Add(len(n.layers))
	for _, l := range n.layers {
		go func(layer layer) {
			defer wg.Done()
			layer.Initialize(n.initializer)
		}(l)
	}
	wg.Wait()

	n.backpropagation(samples)
}

func (n *Network) backpropagation(samples []Sample) {
	iter := 0
	for iter < 50000 {
		if iter%1000 == 0 {
			fmt.Println(iter, MeanSquareError(n, samples))
		}

		preprocess(samples)
		n.completeEpoch(samples)
		iter++
	}
}

func preprocess(samples []Sample) {
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(samples), func(i, j int) { samples[i], samples[j] = samples[j], samples[i] })
}

func (n *Network) completeEpoch(samples []Sample) {
	weightsGradient := constructWeights(n.neurons)
	biasGradient := constructBiases(n.neurons)

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

			for i := 0; i < len(weightsGradient[k]); i++ {
				for j := 0; j < len(weightsGradient[k][i]); j++ {
					weightsGradient[k][i][j] += n.eta * delta[j] * prevLayerOutput[i]
				}
			}
			for i := 0; i < len(biasGradient[k]); i++ {
				biasGradient[k][i] += n.eta * delta[i]
			}

			diff = delta
		}

		var wg sync.WaitGroup
		wg.Add(len(n.layers))
		for k := 0; k < len(n.layers); k++ {
			go func(layer layer, weightsGradient [][]float64, biasGradient []float64) {
				defer wg.Done()
				layer.update(weightsGradient, biasGradient)
			}(n.layers[k], weightsGradient[k], biasGradient[k])
		}
		wg.Wait()

		clearGradients(weightsGradient, biasGradient)
	}
}

func clearGradients(weightsGradient [][][]float64, biasGradient [][]float64) {
	clear := func(array [][]float64, wq *sync.WaitGroup) {
		defer wq.Done()
		for i := 0; i < len(array); i++ {
			for j := 0; j < len(array[i]); j++ {
				array[i][j] = 0.
			}
		}
	}

	var wg sync.WaitGroup
	wg.Add(len(weightsGradient) + 1)
	for k := 0; k < len(weightsGradient); k++ {
		go clear(weightsGradient[k], &wg)
	}
	go clear(biasGradient, &wg)

	wg.Wait()
}

func (n *Network) Predict(input []float64) []float64 {
	output := n.layers[0].processInput(input)
	for i := 1; i < len(n.layers); i++ {
		output = n.layers[i].processInput(output)
	}
	return output
}
