package feedforward

import "sync"

type layer interface {
	Initialize(Initializer)
	processInput([]float64) []float64
	getOutputCache() []float64
	processError([]float64) []float64
	update([][]float64, []float64)
}

type baseLayer struct {
	weights    [][]float64
	biases     []float64
	activation ActivationFunction

	prevLayerNeurons int
	neurons          int

	outputCache []float64
}

func (l *baseLayer) Initialize(initializer Initializer) {
	initializer.Initialize(l.weights)
	for i := 0; i < l.neurons; i++ {
		l.biases[i] = 0
	}
}

func (l *baseLayer) processInput(input []float64) []float64 {
	output := make([]float64, l.neurons)

	var wg sync.WaitGroup
	wg.Add(l.neurons)
	for i := 0; i < l.neurons; i++ {
		go func(i int) {
			defer wg.Done()
			output[i] = l.activation.Value(l.net(i, input))
		}(i)
	}
	wg.Wait()

	l.outputCache = output
	return output
}

func (l *baseLayer) net(i int, input []float64) float64 {
	net := l.biases[i]
	for j := 0; j < l.prevLayerNeurons; j++ {
		net += input[j] * l.weights[j][i]
	}
	return net
}

func (l baseLayer) getOutputCache() []float64 {
	return l.outputCache
}

func (l *baseLayer) update(weights [][]float64, biases []float64) {
	for i := 0; i < l.prevLayerNeurons; i++ {
		for j := 0; j < l.neurons; j++ {
			l.weights[i][j] += weights[i][j]
			l.biases[j] += biases[j]
		}
	}
	l.outputCache = nil
}

type hiddenLayer struct {
	baseLayer
	nextLayerNeurons int
	nextLayerWeights [][]float64
}

func newHiddenLayer(weights [][]float64, biases []float64, nextLayerWeights [][]float64, activation ActivationFunction) layer {
	return &hiddenLayer{
		baseLayer:        baseLayer{weights: weights, biases: biases, activation: activation, prevLayerNeurons: len(weights), neurons: len(biases)},
		nextLayerWeights: nextLayerWeights,
		nextLayerNeurons: len(nextLayerWeights[0]),
	}
}

func (h *hiddenLayer) processError(delta []float64) []float64 {
	layerError := make([]float64, h.neurons)
	output := h.outputCache

	var wg sync.WaitGroup
	wg.Add(h.neurons)
	for i := 0; i < h.neurons; i++ {
		go func(neuron int, weights []float64) {
			defer wg.Done()
			sum := 0.
			for i := 0; i < h.nextLayerNeurons; i++ {
				sum += delta[i] * weights[i]
			}
			layerError[neuron] = h.activation.Gradient(output[neuron]) * sum
		}(i, h.nextLayerWeights[i])
	}
	wg.Wait()

	return layerError
}

type outputLayer struct {
	baseLayer
}

func newOutputLayer(weights [][]float64, biases []float64, activation ActivationFunction) layer {
	return &outputLayer{
		baseLayer: baseLayer{weights: weights, biases: biases, activation: activation, prevLayerNeurons: len(weights), neurons: len(biases)},
	}
}

func (o *outputLayer) processError(delta []float64) []float64 {
	layerError := make([]float64, o.neurons)
	output := o.outputCache
	for i := 0; i < o.neurons; i++ {
		layerError[i] = o.activation.Gradient(output[i]) * delta[i]
	}
	return layerError
}
