from perceptron.linear import Perceptron


# Dataset AND
dataset = [
    ([1, 1], 1),
    ([1, 0], 0),
    ([0, 1], 0),
    ([0, 0], 0)
]


model = Perceptron(n_input=2)


# Training
epochs = 5000
learning_rate = 0.1

for epoch in range(epochs):

    error = 0

    for inputs, target in dataset:
        error += model.train(
            inputs,
            target,
            learning_rate
        )

    if epoch % 1000 == 0:
        print(f"Error: {error / len(dataset)}")


# Test
for inputs, target in dataset:

    prediction = model.forward(inputs)

    print(
        f"Input: {inputs} | "
        f"Target: {target} | "
        f"Prediction: {prediction:.4f}"
    )
