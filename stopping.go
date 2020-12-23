package feedforward

type StoppingCondition struct {
	IsMet func(statistic IterationStatistic) bool
}

func (c StoppingCondition) And(other StoppingCondition) StoppingCondition {
	return StoppingCondition{
		IsMet: func(statistic IterationStatistic) bool { return c.IsMet(statistic) && other.IsMet(statistic) },
	}
}

func (c StoppingCondition) Or(other StoppingCondition) StoppingCondition {
	return StoppingCondition{
		IsMet: func(statistic IterationStatistic) bool { return c.IsMet(statistic) || other.IsMet(statistic) },
	}
}

func (c StoppingCondition) Not() StoppingCondition {
	return StoppingCondition{
		IsMet: func(statistic IterationStatistic) bool { return !c.IsMet(statistic) },
	}
}

func NewMaxIter(maxIter int) StoppingCondition {
	return StoppingCondition{IsMet: func(statistic IterationStatistic) bool { return statistic.iteration >= maxIter }}
}

func NewPrecision(precision float64) StoppingCondition {
	return StoppingCondition{IsMet: func(statistic IterationStatistic) bool { return statistic.score <= precision }}
}
