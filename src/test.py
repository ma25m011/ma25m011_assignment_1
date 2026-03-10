import numpy as np
import argparse
from sklearn.metrics import f1_score
from ann.neural_network import NeuralNetwork

best_config = argparse.Namespace(
    dataset="fashion_mnist",
    epochs=20,
    batch_size=64,
    loss="cross_entropy",
    optimizer="rmsprop",
    weight_decay=0.0,
    learning_rate=0.001,
    num_layers=3,
    hidden_size=[128, 128, 64],
    activation="relu",
    weight_init="xavier",
)

model = NeuralNetwork(best_config)
weights = np.load("best_model.npy", allow_pickle=True)[0]
model.set_weights(weights)

X_test = np.random.rand(100, 784)
y_true = np.random.randint(0, 10, size=(100,))

y_pred = model.forward(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

print("F1 Score:", f1_score(y_true, y_pred_labels, average="macro"))
