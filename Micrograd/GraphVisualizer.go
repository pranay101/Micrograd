package micrograd

import (
	"fmt"
	"os"
)

// VisualizeMLP generates a DOT file for the MLP
func VisualizeGraph(mlp MLP, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Start DOT file content
	fmt.Fprintln(file, "digraph MLP {")
	fmt.Fprintln(file, "  rankdir=LR;")
	fmt.Fprintln(file, "  node [shape=circle];")

	// Layer and neuron visualization
	for layerIdx, layer := range mlp.Layers {
		for neuronIdx, neuron := range layer.Neurons {
			neuronID := fmt.Sprintf("L%dN%d", layerIdx, neuronIdx)
			fmt.Fprintf(file, "  %s [label=\"Neuron\"];\n", neuronID)
			for weightIdx, weight := range neuron.W {
				inputID := fmt.Sprintf("L%dN%dW%d", layerIdx-1, neuronIdx, weightIdx)
				fmt.Fprintf(file, "  %s -> %s [label=\"%.2f\"];\n", inputID, neuronID, weight.Val)
			}
			fmt.Fprintf(file, "  bias%d -> %s [label=\"%.2f\"];\n", neuronIdx, neuronID, neuron.B.Val)
		}
	}

	// End DOT file content
	fmt.Fprintln(file, "}")
	return nil
}
