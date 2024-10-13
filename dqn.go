package main

import (
	"flag"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"syscall"

	"net/http"
	_ "net/http/pprof"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	epochs     = flag.Int("epochs", 100, "Number of epochs to train for")
	batchsize  = flag.Int("batchsize", 100, "Batch size")
	cpuprofile = flag.String("cpuprofile", "", "CPU profiling")
)

const boardDepth = 12
const boardSize = 8

var dt tensor.Dtype

func parseDtype() {
	dt = tensor.Float32 // Use float32 for the DQN
}

type dqn struct {
	g          *G.ExprGraph
	w1, w2, w3 *G.Node // weights for layers
	out        *G.Node
}

func newDQN(g *G.ExprGraph) *dqn {
	w1 := G.NewTensor(g, dt, 4, G.WithShape(32, boardDepth, 3, 3), G.WithName("w1"), G.WithInit(G.GlorotN(1.0)))
	w2 := G.NewTensor(g, dt, 4, G.WithShape(64, 32, 3, 3), G.WithName("w2"), G.WithInit(G.GlorotN(1.0)))
	w3 := G.NewMatrix(g, dt, G.WithShape(64*2*2, 1876), G.WithName("w3"), G.WithInit(G.GlorotN(1.0))) // Example: 1876 possible moves
	return &dqn{
		g:  g,
		w1: w1,
		w2: w2,
		w3: w3,
	}
}

func (m *dqn) fwd(x *G.Node) (err error) {
	var c1, c2, fc *G.Node
	var a1, a2 *G.Node

	// Convolution Layer 1
	if c1, err = G.Conv2d(x, m.w1, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return err
	}
	if a1, err = G.Rectify(c1); err != nil {
		return err
	}

	// Max Pooling Layer 1
	if a1, err = G.MaxPool2D(a1, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		return err
	}

	// Convolution Layer 2
	if c2, err = G.Conv2d(a1, m.w2, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return err
	}
	if a2, err = G.Rectify(c2); err != nil {
		return err
	}

	// Flattening
	b, c, h, w := a2.Shape()[0], a2.Shape()[1], a2.Shape()[2], a2.Shape()[3]
	r2, err := G.Reshape(a2, tensor.Shape{b, c * h * w})
	if err != nil {
		return err
	}

	// Fully Connected Layer
	if fc, err = G.Mul(r2, m.w3); err != nil {
		return err
	}

	// Output Q-values
	m.out, err = G.Rectify(fc)
	return err
}

func DQNmain() {
	flag.Parse()
	parseDtype()
	rand.Seed(1337)

	// intercept Ctrl+C
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	doneChan := make(chan bool, 1)

	go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()

	g := G.NewGraph()
	x := G.NewTensor(g, dt, 4, G.WithShape(-1, boardDepth, boardSize, boardSize), G.WithName("x"))
	m := newDQN(g)
	if err := m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	// Define the loss function
	target := G.NewMatrix(g, dt, G.WithShape(-1, 1876), G.WithName("target"))
	loss := G.Must(G.Square(G.Must(G.Sub(m.out, target))))
	cost := G.Must(G.Mean(loss))
	cost = G.Must(G.Neg(cost))

	// Gradients
	if _, err := G.Grad(cost, m.w1, m.w2, m.w3); err != nil {
		log.Fatal(err)
	}

	prog, locMap, _ := G.Compile(g)
	vm := G.NewTapeMachine(g, G.WithPrecompiled(prog, locMap), G.BindDualValues(m.w1, m.w2, m.w3))
	solver := G.NewRMSPropSolver(G.WithBatchSize(float64(*batchsize)))
	defer vm.Close()

	// Main training loop
	for i := 0; i < *epochs; i++ {
		// Reset the VM for each epoch
		vm.Reset()

		xBatch, targetBatch := getBatchData() // Get positions from k3s pod, should include stockfish and in house rating for each move to train on

		G.Let(x, xBatch)
		G.Let(target, targetBatch)

		// Run the forward pass
		if err := vm.RunAll(); err != nil {
			log.Fatalf("Failed to run VM at epoch %d: %v", i, err)
		}

		// Step the solver to update weights
		if err := solver.Step(G.NodesToValueGrads(G.Nodes{m.w1, m.w2, m.w3})); err != nil {
			log.Fatalf("Failed to update nodes with gradients at epoch %d: %v", i, err)
		}

		// Optionally, log the cost for monitoring
		var costVal G.Value
		G.Read(cost, &costVal)
		log.Printf("Epoch %d | Cost: %v", i, costVal)

		// Reset the VM for the next iteration
		vm.Reset()
	}

	// Cleanup
	cleanup(sigChan, doneChan)
}

func cleanup(sigChan chan os.Signal, doneChan chan bool) {
	select {
	case <-sigChan:
		log.Println("EMERGENCY EXIT!")
		os.Exit(1)
	case <-doneChan:
		return
	}
}
