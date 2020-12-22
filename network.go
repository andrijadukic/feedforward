package feedforward

import (
	"math/rand"
	"time"
)

type Model interface {
	Fit([]Sample)
	Predict([]float64) []float64
}

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
		biases[k] = make([]float64, neurons[k])
	}
	return biases
}

func (n *Network) Fit(samples []Sample) {
	for _, layer := range n.layers {
		layer.Initialize(n.initializer)
	}

	n.backpropagation(samples)
}

func (n *Network) backpropagation(samples []Sample) {
	iter := 0
	for iter < 1000 {

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
	for _, sample := range samples {
		weightsGradient := constructWeights(n.neurons)
		biasGradient := constructBiases(n.neurons)

		input := sample.Input
		expected := sample.Output
		actual := n.Predict(input)
		diff := make([]float64, len(expected))
		for i := 0; i < len(diff); i++ {
			diff[i] = expected[i] - actual[i]
		}

		for k := len(n.layers) - 1; k > 0; k-- {
			delta := n.layers[k].processError(diff)
			prevLayerOutput := n.layers[k-1].getOutputCache()

			for i := 0; i < len(weightsGradient[k]); i++ {
				for j := 0; j < len(weightsGradient[k][i]); j++ {
					weightsGradient[k][i][j] += delta[j] * prevLayerOutput[i]
				}
			}
			for i := 0; i < len(biasGradient[k]); i++ {
				biasGradient[k][i] += delta[i]
			}
		}

		for k := 0; k < len(weightsGradient); k++ {
			for i := 0; i < len(weightsGradient[k]); i++ {
				for j := 0; j < len(weightsGradient[k][i]); j++ {
					weightsGradient[k][i][j] *= n.eta
				}
			}
		}
		for k := 0; k < len(biasGradient); k++ {
			for i := 0; i < len(biasGradient[k]); i++ {
				biasGradient[k][i] *= n.eta
			}
		}

		for k := 0; k < len(n.layers); k++ {
			n.layers[k].update(weightsGradient[k], biasGradient[k])
		}
	}
}

func (n *Network) Predict(input []float64) []float64 {
	output := n.layers[0].processInput(input)
	for i := 1; i < len(n.layers); i++ {
		output = n.layers[i].processInput(output)
	}
	return output
}
