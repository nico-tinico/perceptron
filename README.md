# Perceptron

A collection of neural network implementations from scratch in Python.

The goal of this repository is to explore and implement different types of perceptrons and neural network architectures, starting from simple linear models and progressively moving towards more complex multilayer networks.

## Project Structure

The repository is organized into different architectures, each with its own implementation.

```text
perceptron/
├── src/
│   └── perceptron/
│       ├── activations/
│       ├── losses/
│       ├── linear/
│       └── multi/
│
├── examples/
│   ├── linear/
│   └── multi/
│
├── tests/
│
├── pyproject.toml
├── README.md
└── .gitignore
```

## Architectures

### Linear

Contains single-neuron perceptron implementations based on linear decision boundaries.

Currently includes:

- Sigmoid activation
- Mean Squared Error (MSE)
- Gradient Descent optimization

### Multi

Dedicated to multilayer perceptrons (MLP) and more complex neural network architectures.

Planned implementations include:

- Layers
- Forward propagation
- Backpropagation
- Multilayer training

## Shared Components

The `activations` and `losses` packages contain reusable mathematical functions shared across the different architectures.

This approach avoids code duplication and allows different models to reuse common components.

## Installation

Clone the repository:

```bash
git clone https://github.com/nico-tinico/perceptron.git
cd perceptron
```

Install the project in editable mode:

```bash
python -m pip install -e .
```

## Examples

Examples are organized by architecture.

To run the linear AND perceptron:

```bash
python examples/linear/and.py
```

## Development

The project is developed using Python 3.11+ and aims to keep implementations simple, modular and dependency-free.

## Project status

This repository is a work in progress. Some elements are missing; specifically, the files containing only the code necessary for the operation of a linear perceptron based on MSE and the sigmoid function.
