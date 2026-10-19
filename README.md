# 🧠 NeuroViz — Neural Network Playground

**See the math. Understand the magic. Build your intuition.**

NeuroViz is an interactive neural network visualizer designed to make the inner workings of neural networks easier to understand. Build a network, experiment with its parameters, and watch the calculations happen in real time.

Instead of treating a neural network as a black box, NeuroViz breaks it down into small, understandable steps.

### ✨ Features

- **Interactive Network Builder** — Adjust the number of input neurons, hidden layers, and neurons per layer.
- **Visual Neural Network** — Explore neurons and the weighted connections between them.
- **Editable Parameters** — Change input values, weights, and biases to see how they affect the output.
- **Activation Functions** — Experiment with Sigmoid, ReLU, and Tanh.
- **Live Calculations** — Observe how changes propagate through the network during forward propagation.
- **Step-by-Step Explanations** — Follow the mathematics behind weighted sums, biases, activation functions, and the final output.

### 🧮 How It Works

A neural network processes information through interconnected layers.

1. **Input Layer:** Receives the input values.
2. **Hidden Layers:** Each neuron calculates a weighted sum of its inputs, adds a bias, and applies an activation function.
3. **Output Layer:** Produces the final output based on the preceding layer's activations.

The core calculation for a neuron is:
z = \sum_{i=1}^{n} wi * xi + b
\]

\[
a = f(z)
\]

Where:

- \(x_i\) — input values
- \(w_i\) — connection weights
- \(b\) — bias
- \(z\) — weighted sum before activation
- \(f\) — activation function
- \(a\) — neuron's output or activation

NeuroViz lets you experiment with these values and see how each calculation contributes to the final result.

### ⚡ Activation Functions

| Function    | Purpose                                                         |
|-------------|-----------------------------------------------------------------|
| **Sigmoid** | Maps values to a range between 0 and 1                          |
| **ReLU**    | Returns 0 for negative inputs, input itself for positive inputs |
| **Tanh**    | Maps values to a range between -1 and 1                         |

### 🛠️ Tech Stack

**Frontend**
- HTML5
- CSS3
- JavaScript
- SVG for network visualization

**Backend**
- Python
- FastAPI

**Calculations**
- JavaScript for interactive frontend calculations
- Python API for backend calculations

### 🚀 Getting Started

#### Prerequisites

- Python 3.10 or newer
- A modern web browser
- VS Code (recommended)

#### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd neuroviz_desmos
```

Replace `YOUR_REPOSITORY_URL` with your GitHub repository URL.

#### 2. Set up the Python backend

Create a virtual environment:

```bash
py -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

#### 3. Start the backend

```bash
python -m uvicorn backend.main:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

#### 4. Open the frontend

Open `index.html` in your browser, or use the VS Code Live Server extension.

> **Note:** In the current prototype, the frontend performs its interactive calculations in JavaScript, while the FastAPI backend provides a separate calculation endpoint. The frontend and backend are not yet connected.

### 🔮 Future Improvements

- Train networks using backpropagation and gradient descent.
- Visualize how weights change during training.
- Add datasets and training controls.
- Display activation values directly on neurons.
- Support additional network architectures and activation functions.

### 🎯 Project Goal

NeuroViz aims to make neural networks less intimidating by turning abstract equations into an interactive visual learning experience.

Built for curious minds who want to understand not just *what* a neural network predicts, but *how* it gets there.

---

**Made with curiosity and a love for machine learning.** 💚