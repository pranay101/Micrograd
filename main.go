package main

import (
	"Micrograd/Micrograd"
)

func main() {
	mlp := micrograd.NewMLP(2, []int{2, 2})
	micrograd.VisualizeGraph(mlp, "mlp.dot")
}
