# gofeedforward

A multilayer feedforward neural network implemented in Go, trained via backpropagation.  
The purpose of this package is threefold:

- Experiment with building things in Go
- Provide a working implementation of a neural network algorithm that most people start of with
- Provide a useful starting point for a more advanced, optimized implementation

It is not a goal to provide the fastest possible implementation, as those already exist and are inherently not as
readable, especially for people only starting out.

Recommendations are very welcome :)

## Usage

``` go
...
samples, err := feedforward.Load(path, feedforward.Delimiters{"," ," -> ", ","})
if err != nil {
    panic(err)
}

neural := feedforward.NewNetwork(
    []int{40, 40, 5},
    []feedforward.ActivationFunction{feedforward.Sigmoid(), feedforward.Sigmoid(), feedforward.Sigmoid()},
    feedforward.NewUniformInitializer(-1, 1),
    feedforward.NewMaxIter(10000).Or(feedforward.NewPrecision(1e-5)),
    0.1)

neural.AddObserver(feedforward.NewNthIterationObserver(feedforward.NewStOutLogger(), 1000))

neural.Fit(samples)

prediction, err := neural.Predict(samples[0].Input)
if err != nil {
    panic(err)
}

fmt.Println("Prediction:", prediction)
fmt.Println("Actual:", samples[0].Output)
...
```