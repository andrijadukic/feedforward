package feedforward

import "fmt"

// Type which holds information about the current iteration of an iterative algorithm.
// iteration holds the current iteration count.
// score holds a floating point value representing a loss function score (lower is better).
type IterationStatistic struct {
	iteration int
	score     float64
}

// Interface defining an observer of an iterative process.
type ModelObserver interface {
	Update(statistic IterationStatistic)
}

// Interface defining a publisher of iteration statistics.
// It provides methods for adding, removing and notifying any number of observers
type ModelSubject interface {
	AddObserver(observer ModelObserver)
	RemoveObserver(observer ModelObserver)
	NotifyObservers(statistic IterationStatistic)
}

// Struct implementing the entire ModelSubject interface.
// Extended by types which publish iteration statistics.
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
// to receive an update only every nth iteration.
type nthIterObserver struct {
	observer ModelObserver
	iter     int
}

// Constructor for generating a new nthIterObserver
func NewNthIterationObserver(observer ModelObserver, iter int) ModelObserver {
	return &nthIterObserver{observer: observer, iter: iter}
}

// Checks if current iteration is divisible by iter, forwards the given statistic to the underlying observer if true
func (s *nthIterObserver) Update(statistic IterationStatistic) {
	if statistic.iteration%s.iter == 0 {
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
	fmt.Println(statistic.iteration, statistic.score)
}
