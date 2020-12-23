package feedforward

import "fmt"

type IterationStatistic struct {
	iteration int
	score     float64
}

type ModelObserver interface {
	Update(statistic IterationStatistic)
}

type ModelSubject interface {
	AddObserver(observer ModelObserver)
	RemoveObserver(observer ModelObserver)
	NotifyObservers(statistic IterationStatistic)
}

type BaseSubject struct {
	observers []ModelObserver
}

func (s *BaseSubject) AddObserver(observer ModelObserver) {
	s.observers = append(s.observers, observer)
}

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

func (s *BaseSubject) NotifyObservers(statistic IterationStatistic) {
	for _, observer := range s.observers {
		observer.Update(statistic)
	}
}

type nthIterObserver struct {
	observer ModelObserver
	iter     int
}

func NewNthIterationObserver(observer ModelObserver, iter int) ModelObserver {
	return &nthIterObserver{observer: observer, iter: iter}
}

func (s *nthIterObserver) Update(statistic IterationStatistic) {
	if statistic.iteration%s.iter == 0 {
		s.observer.Update(statistic)
	}
}

type stoutLogger struct{}

func NewStOutLogger() ModelObserver {
	return &stoutLogger{}
}

func (s *stoutLogger) Update(statistic IterationStatistic) {
	fmt.Println(statistic.iteration, statistic.score)
}
