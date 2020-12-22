package feedforward

func main() {
	model := NewNetwork([]int{2, 2, 2}, []ActivationFunction{Sigmoid{}, Sigmoid{}}, NewUniformInitializer(-1, 1), 0.1)
	model.Fit(nil)
}
