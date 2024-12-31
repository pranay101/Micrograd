package micrograd

import (
	"math"
	"math/rand"
	"time"
)

// Value represents a single value with basic operations
type Value struct {
	Val  float64
	Grad float64 // For backpropagation
	Name string  // Optional name for visualization
}

// NewValue creates a new Value
func NewValue(v float64, name string) Value {
	return Value{Val: v, Grad: 0.0, Name: name}
}

// Tanh activation function
func (v Value) Tanh() Value {
	return NewValue(math.Tanh(v.Val), v.Name+"_tanh")
}

// Neuron represents a single neuron
type Neuron struct {
	W []Value // Weights
	B Value   // Bias
}

// NewNeuron initializes a neuron with random weights and bias
func NewNeuron(numInputs int) Neuron {
	rand.Seed(time.Now().UnixNano())
	weights := make([]Value, numInputs)
	for i := 0; i < numInputs; i++ {
		weights[i] = NewValue(rand.Float64()*2-1, "w"+ string(i))
	}
	bias := NewValue(rand.Float64()*2-1, "b")
	return Neuron{W: weights, B: bias}
}

// Call performs the forward computation for a neuron
func (n Neuron) Call(x []Value) Value {
	sum := n.B
	for i, wi := range n.W {
		sum.Val += wi.Val * x[i].Val
	}
	return sum.Tanh()
}

// Parameters returns all parameters of the neuron
func (n Neuron) Parameters() []Value {
	return append(n.W, n.B)
}

// Layer represents a layer of neurons
type Layer struct {
	Neurons []Neuron
}

// NewLayer initializes a layer with the specified number of neurons
func NewLayer(numInputs, numNeurons int) Layer {
	neurons := make([]Neuron, numNeurons)
	for i := 0; i < numNeurons; i++ {
		neurons[i] = NewNeuron(numInputs)
	}
	return Layer{Neurons: neurons}
}

// Call performs the forward computation for a layer
func (l Layer) Call(x []Value) []Value {
	outputs := make([]Value, len(l.Neurons))
	for i, neuron := range l.Neurons {
		outputs[i] = neuron.Call(x)
	}
	return outputs
}

// Parameters returns all parameters of the layer
func (l Layer) Parameters() []Value {
	var params []Value
	for _, neuron := range l.Neurons {
		params = append(params, neuron.Parameters()...)
	}
	return params
}

// MLP represents a multi-layer perceptron
type MLP struct {
	Layers []Layer
}

// NewMLP initializes an MLP with the specified layers
func NewMLP(numInputs int, layerSizes []int) MLP {
	sizes := append([]int{numInputs}, layerSizes...)
	layers := make([]Layer, len(layerSizes))
	for i := 0; i < len(layerSizes); i++ {
		layers[i] = NewLayer(sizes[i], sizes[i+1])
	}
	return MLP{Layers: layers}
}

// Call performs the forward computation for the MLP
func (m MLP) Call(x []Value) []Value {
	for _, layer := range m.Layers {
		x = layer.Call(x)
	}
	return x
}

// Parameters returns all parameters of the MLP
func (m MLP) Parameters() []Value {
	var params []Value
	for _, layer := range m.Layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}
