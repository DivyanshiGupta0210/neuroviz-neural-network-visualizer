from math import exp, tanh
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="NeuroViz API", description="Forward-pass calculations for the NeuroViz playground.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Local learning project only; restrict this before deployment.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CalculationRequest(BaseModel):
    inputs: list[float] = Field(min_length=1, max_length=4)
    weights: list[list[list[float]]]
    biases: list[list[float]]
    activation: str = "sigmoid"


def activate(value: float, name: str) -> float:
    if name == "relu":
        return max(0.0, value)
    if name == "tanh":
        return tanh(value)
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + exp(-value))


@app.get("/")
def home():
    return {"message": "NeuroViz API is running", "docs": "/docs"}


@app.post("/calculate")
def calculate(data: CalculationRequest):
    """Calculate a forward pass. weights[layer][from_neuron][to_neuron]."""
    if data.activation not in {"sigmoid", "relu", "tanh"}:
        return {"error": "Activation must be sigmoid, relu, or tanh."}

    current = data.inputs[:]
    layer_details = []
    for layer_index, matrix in enumerate(data.weights):
        next_values = []
        layer_steps = []
        for to_index, column in enumerate(zip(*matrix)):
            bias = data.biases[layer_index][to_index]
            contributions = [
                {"input": current[i], "weight": column[i],
                 "contribution": current[i] * column[i]}
                for i in range(len(current))
            ]
            total = sum(item["contribution"] for item in contributions) + bias
            output = activate(total, data.activation)
            next_values.append(output)
            layer_steps.append({
                "neuron": to_index,
                "contributions": contributions,
                "bias": bias,
                "weighted_total": total,
                "activation": data.activation,
                "output": output,
            })
        layer_details.append({"layer": layer_index + 1, "neurons": layer_steps})
        current = next_values

    return {"layers": layer_details, "output": current}


# Run from the project root:
# python -m uvicorn backend.main:app --reload
