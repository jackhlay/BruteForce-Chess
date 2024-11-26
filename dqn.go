package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"log"
	"math"
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

type Data struct {
	StartFEN    *tensor.Dense
	StartRating float64
	Action      string
	EndFEN      *tensor.Dense
	EndRating   float64
}

type RequestData struct {
	Startfen    string  `json:"startFEN"`
	StartRating float64 `json:"startRating"`
	Endfen      string  `json:"endFEN"`
	EndRating   float64 `json:"EndRating"`
	Action      string  `json:"action"`
}

var trainingData []Data

var actionMap map[string]int // Load with Legal moves every time called.

var epsilon = 0.1 // Exploration rate

const boardDepth = 12
const boardSize = 8

var dt tensor.Dtype
var costVal G.Value

func parseDtype() {
	dt = tensor.Float32
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
		log.Printf("Epoch %d | Cost: %v | Epsilon: %v", i, costVal, epsilon)
		if i%10 == 0 {
			epsilon = math.Max(0.01, epsilon*0.99) // Decay epsilon
		}

		// Reset the VM for each epoch
		vm.Reset()

		xBatch, targetBatch := getBatchData(trainingData) //the data coming in is turned into appropriate types (Tensor, float, string, Tensor, float), and into Data slice

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

func chooseAction(qValues []float32) int {
	if rand.Float64() < epsilon {
		return rand.Intn(len(qValues)) // Random action
	}
	maxIdx := 0
	for i, v := range qValues {
		if v > qValues[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}

func getActionIndex(action string) int {
	if idx, exists := actionMap[action]; exists {
		return idx
	}
	log.Fatalf("Action %s not found in action map", action)
	return -1
}

func getBatchData(tData []Data) (*tensor.Dense, *tensor.Dense) {
	batchSize := len(tData)

	xBacking := make([]float32, batchSize*boardDepth*boardSize*boardSize)
	targetBacking := make([]float32, batchSize*1876)

	xBatch := tensor.New(tensor.WithShape(batchSize, boardDepth, boardSize, boardSize), tensor.WithBacking(xBacking))
	targetBatch := tensor.New(tensor.WithShape(batchSize, 1876), tensor.WithBacking(targetBacking))

	for i, data := range tData {
		// Copy StartFEN tensor data into xBatch
		startFENData := data.StartFEN.Data().([]float32)
		copy(xBacking[i*boardDepth*boardSize*boardSize:], startFENData)

		// Calculate reward
		reward := data.EndRating - data.StartRating

		// Update Q-values
		actionIndex := getActionIndex(data.Action)
		targetBacking[i*1876+actionIndex] = float32(reward)
	}

	return xBatch, targetBatch
}

func handleIncomingJSON(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}

	var data RequestData
	err := json.NewDecoder(r.Body).Decode(&data)
	if err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	startPlane, _ := makePlanes(data.Startfen)
	endPlane, _ := makePlanes(data.Endfen)
	Tdata := Data{
		StartFEN:    startPlane,
		StartRating: data.StartRating,
		Action:      data.Action,
		EndFEN:      endPlane,
		EndRating:   data.EndRating,
	}
	trainingData = append(trainingData, Tdata)

	w.WriteHeader(http.StatusOK)
}

func saveModel(filename string, weights ...*G.Node) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	for _, weight := range weights {
		w := weight.Value().Data().([]float32)
		if err := binary.Write(file, binary.LittleEndian, w); err != nil {
			return err
		}
	}
	return nil
}

func loadModel(filename string, weights ...*G.Node) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	for _, weight := range weights {
		w := weight.Value().Data().([]float32)
		if err := binary.Read(file, binary.LittleEndian, w); err != nil {
			return err
		}
	}
	return nil
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

// ENDPOINT CODE

func handlePredict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}

	var data struct {
		StartFEN string `json:"startFEN"`
	}
	err := json.NewDecoder(r.Body).Decode(&data)
	if err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	g := G.NewGraph()
	m := newDQN(g)

	startPlane, _ := makePlanes(data.StartFEN)
	x := tensor.New(tensor.WithShape(1, boardDepth, boardSize, boardSize), tensor.WithBacking(startPlane.Data()))

	// Create the computation graph node using the tensor
	node := G.NewTensor(g, dt, 4, G.WithShape(1, boardDepth, boardSize, boardSize), G.WithValue(x))

	// Perform the forward pass with the created node
	if err := m.fwd(node); err != nil {
		http.Error(w, "Model forward pass failed", http.StatusInternalServerError)
		return
	}

	// Get Q-values and pick the best action
	qValues := m.out.Value().Data().([]float32)
	bestActionIdx := chooseAction(qValues)

	for action, idx := range actionMap {
		if idx == bestActionIdx {
			json.NewEncoder(w).Encode(map[string]string{"bestMove": action})
			return
		}
	}

	http.Error(w, "Failed to determine best move", http.StatusInternalServerError)
}

func listen() {
	http.HandleFunc("/your-endpoint", handleIncomingJSON) // Define your route here
	http.HandleFunc("/predict", handlePredict)            // Define your route here

	log.Println("Starting server on :5000...")
	if err := http.ListenAndServe(":5000", nil); err != nil {
		log.Fatal(err)
	}
}
