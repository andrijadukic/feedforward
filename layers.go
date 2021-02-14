package feedforward

// Represents a layer of a feedforward neural network.
type layer interface {
	initialize(Initializer)
	processInput([]float64) []float64
	getOutputCache() []float64
	processError([]float64) []float64
	getWeights() [][]float64
	getBiases() []float64
}

// Base layer implementation
type baseLayer struct {
	weights    [][]float64
	biases     []float64
	activation ActivationFunction

	prevLayerNeurons int
	neurons          int

	outputCache []float64
}

// Initializes weights and biases of the entire layer using the provided initializer
func (l *baseLayer) initialize(initializer Initializer) {
	initializer.Initialize(l.weights)
	for i := 0; i < l.neurons; i++ {
		l.biases[i] = 0
	}
}

// Computes output of the entire layer for the given input and caches the output before returning to caller.
// The output is computed concurrently for each neuron through goroutines, synchronized through a WaitGroup.
func (l *baseLayer) processInput(input []float64) []float64 {
	output := make([]float64, l.neurons)

	for i := 0; i < l.neurons; i++ {
		output[i] = l.activation.Value(l.net(i, input))
	}

	l.outputCache = output
	return output
}

// Computes net of i-th neuron
func (l *baseLayer) net(i int, input []float64) float64 {
	net := l.biases[i]
	for j := 0; j < l.prevLayerNeurons; j++ {
		net += input[j] * l.weights[j][i]
	}
	return net
}

// Gets cached output
func (l *baseLayer) getOutputCache() []float64 {
	return l.outputCache
}

// Gets underlying weight slice
func (l *baseLayer) getWeights() [][]float64 {
	return l.weights
}

// Gets underlying bias slice
func (l *baseLayer) getBiases() []float64 {
	return l.biases
}

// Type representing a hidden layer.
// Extends all properties from the baseLayer.
// Additionally holds outgoing weights used in calculating the weighted layer error.
type hiddenLayer struct {
	baseLayer
	nextLayerNeurons int
	nextLayerWeights [][]float64
}

// Constructor of a hidden layer.
func newHiddenLayer(weights [][]float64, biases []float64, nextLayerWeights [][]float64, activation ActivationFunction) layer {
	return &hiddenLayer{
		baseLayer:        baseLayer{weights: weights, biases: biases, activation: activation, prevLayerNeurons: len(weights), neurons: len(biases)},
		nextLayerWeights: nextLayerWeights,
		nextLayerNeurons: len(nextLayerWeights[0]),
	}
}

// Computes the weighted error of this layer.
// Computations are performed concurrently for every neuron of this layer.
func (h *hiddenLayer) processError(delta []float64) []float64 {
	layerError := make([]float64, h.neurons)
	output := h.outputCache

	for i := 0; i < h.neurons; i++ {
		sum := 0.
		for j := 0; j < h.nextLayerNeurons; j++ {
			sum += delta[j] * h.nextLayerWeights[i][j]
		}
		layerError[i] = h.activation.Gradient(output[i]) * sum
	}

	return layerError
}

// Type representing a hidden layer.
// Extends all properties from the baseLayer and provides implementation of processError.
type outputLayer struct {
	baseLayer
}

// Constructor of an output layer.
func newOutputLayer(weights [][]float64, biases []float64, activation ActivationFunction) layer {
	return &outputLayer{
		baseLayer: baseLayer{weights: weights, biases: biases, activation: activation, prevLayerNeurons: len(weights), neurons: len(biases)},
	}
}

// Computes the error of the output layer.
func (o *outputLayer) processError(delta []float64) []float64 {
	layerError := make([]float64, o.neurons)
	output := o.outputCache
	for i := 0; i < o.neurons; i++ {
		layerError[i] = o.activation.Gradient(output[i]) * delta[i]
	}
	return layerError
}
