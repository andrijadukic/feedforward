package feedforward

// Struct which models an iterative algorithms stopping condition.
// It holds a single function which takes an IterationStatistics type and returns true if stop condition is met, false otherwise
type StoppingCondition struct {
	IsMet func(statistic IterationStatistic) bool
}

// Method used for combining two stopping conditions into a new stopping condition which will only return true if both
// of the underlying stopping conditions return true
func (c StoppingCondition) And(other StoppingCondition) StoppingCondition {
	return StoppingCondition{
		IsMet: func(statistic IterationStatistic) bool { return c.IsMet(statistic) && other.IsMet(statistic) },
	}
}

// Method used for combining two stopping conditions into a new stopping condition which will return true if either
// of the underlying stopping conditions return true
func (c StoppingCondition) Or(other StoppingCondition) StoppingCondition {
	return StoppingCondition{
		IsMet: func(statistic IterationStatistic) bool { return c.IsMet(statistic) || other.IsMet(statistic) },
	}
}

// Method for inverting a stopping condition.
// Returns a new stopping condition which will return true if original condition returns false and vice versa
func (c StoppingCondition) Not() StoppingCondition {
	return StoppingCondition{
		IsMet: func(statistic IterationStatistic) bool { return !c.IsMet(statistic) },
	}
}

// Returns a new stopping condition which will return true when maximum iteration count is reached
func NewMaxIter(maxIter int) StoppingCondition {
	return StoppingCondition{IsMet: func(statistic IterationStatistic) bool { return statistic.iteration >= maxIter }}
}

// Returns a new stopping condition which will return true when desired precision is reached
func NewPrecision(precision float64) StoppingCondition {
	return StoppingCondition{IsMet: func(statistic IterationStatistic) bool { return statistic.score <= precision }}
}
