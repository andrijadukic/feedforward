package feedforward

import (
	"fmt"
	"math"
)

// Represents a type which holds information about the current Iteration of an iterative algorithm.
// GetIteration returns the current Iteration number.
// GetScore returns a loss function score (lower is better).
type IterationStatistic interface {
	GetIteration() int
	GetScore() float64
}

// Type representing a function which returns a loss function score
type Scorer func() float64

// Implementation of IterationStatistic which holds the current iteration number
// and a Scorer type which computes the loss function score on demand.
// The reason for using a Scorer instead of storing the score directly is to enable lazy evaluation as scores can be
// expensive to compute and not all usages of IterationStatistic call the GetScore method.
// On the first call of GetScore, the computed value will be cached in scoreCache variable.
type iterationStatistic struct {
	iteration  int
	scorer     Scorer
	scoreCache float64
}

// Constructor for a new iterationStatistic, scoreCache is initially set to math.Nan.
func NewIterationStatistic(iteration int, scorer Scorer) IterationStatistic {
	return &iterationStatistic{iteration: iteration, scorer: scorer, scoreCache: math.NaN()}
}

// Gets the iteration number of this iterationStatistic.
func (i *iterationStatistic) GetIteration() int {
	return i.iteration
}

// Gets the score of this iterationStatistic.
// On first method call, the method calls the Scorer type and stores the computed value in scoreCache.
func (i *iterationStatistic) GetScore() float64 {
	if !(i.scoreCache == math.NaN()) {
		i.scoreCache = i.scorer()
	}
	return i.scoreCache
}

// Interface defining an observer of an iterative process.
type ModelObserver interface {
	Update(statistic IterationStatistic)
}

// Interface defining a publisher of Iteration statistics.
// It provides methods for adding, removing and notifying any number of observers
type ModelSubject interface {
	AddObserver(observer ModelObserver)
	RemoveObserver(observer ModelObserver)
	NotifyObservers(statistic IterationStatistic)
}

// Struct implementing the entire ModelSubject interface.
// Extended by types which publish Iteration statistics.
type BaseSubject struct {
	observers []ModelObserver
}

// Adds an observer into a slice of observers.
func (s *BaseSubject) AddObserver(observer ModelObserver) {
	s.observers = append(s.observers, observer)
}

// Removes an observer from the slice of observers.
func (s *BaseSubject) RemoveObserver(observer ModelObserver) {
	index := -1
	for i, attached := range s.observers {
		if attached == observer {
			index = i
		}
	}
	s.observers[index], s.observers[len(s.observers)-1] = s.observers[len(s.observers)-1], s.observers[index]
	s.observers = s.observers[0:len(s.observers)]
}

// Notifies all observers by iterating over the slice and calling Update method of every observer.
func (s *BaseSubject) NotifyObservers(statistic IterationStatistic) {
	for _, observer := range s.observers {
		observer.Update(statistic)
	}
}

// Observer type which serves as a wrapper for a given ModelObserver.
// It calls the Update method of the underlying observer every iter iterations, effectively allowing the underlying observer
// to receive an update only every nth Iteration.
type nthIterObserver struct {
	observer ModelObserver
	iter     int
}

// Constructor for generating a new nthIterObserver
func NewNthIterationObserver(observer ModelObserver, iter int) ModelObserver {
	return &nthIterObserver{observer: observer, iter: iter}
}

// Checks if current Iteration is divisible by iter, forwards the given statistic to the underlying observer if true
func (s *nthIterObserver) Update(statistic IterationStatistic) {
	if statistic.GetIteration()%s.iter == 0 {
		s.observer.Update(statistic)
	}
}

// Observer type which prints to standard output every statistic it receives
type stoutLogger struct{}

// Constructor for generating a new stoutLogger
func NewStOutLogger() ModelObserver {
	return &stoutLogger{}
}

// Prints the given statistic to the standard output
func (s *stoutLogger) Update(statistic IterationStatistic) {
	fmt.Println(statistic.GetIteration(), statistic.GetScore())
}
