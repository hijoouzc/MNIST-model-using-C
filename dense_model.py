"""
train_mnist.py
~~~~~~~~~~~~
Train and evaluate the MNIST feedforward network defined in network2.py, reading data directly from CSV files.
"""
import sys
import os
import json
import random
import numpy as np
from network import Network, CrossEntropyCost, QuadraticCost


def vectorized_result(j):
    """Convert digit j into a 10-dimensional one-hot column vector."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def load_data_from_csv(filename, one_hot=True):
    """
    Load MNIST data from a CSV file.
    CSV format: label,pixel1,...,pixel784
    Returns a list of tuples (input_column, label_or_vector).
    Raises SystemExit if file not found or invalid.
    """
    if not os.path.isfile(filename):
        print(f"Error: File '{filename}' not found. Please ensure the MNIST CSV file is in the working directory.")
        sys.exit(1)
    try:
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
    except Exception as e:
        print(f"Error loading '{filename}': {e}")
        sys.exit(1)

    # Normalize pixel values to [0,1]
    images = data[:, 1:]
    inputs = [img.reshape(784, 1) for img in images]

    labels = data[:, 0].astype(int)
    if one_hot:
        targets = [vectorized_result(y) for y in labels]
    else:
        targets = [y for y in labels]

    return list(zip(inputs, targets))


def main():
    # Load training and test data from CSV
    training_data = load_data_from_csv("mnist_train.csv", one_hot=True)
    full_test_data = load_data_from_csv("mnist_test.csv", one_hot=False)

    # Split out validation set from the beginning of test data
    validation_data = full_test_data[:5000]
    test_data = full_test_data[5000:]

    # Define network architecture
    sizes = [784, 32, 16,  10]
    cost = CrossEntropyCost  # or QuadraticCost

    # Initialize network
    net = Network(sizes, cost=cost)

    # Hyperparameters
    epochs = 20
    mini_batch_size = 60
    eta = 0.02     # learning rate
    lmbda = 5.0   # regularization parameter

    # Train with monitoring on training and validation sets
    eval_cost, eval_acc, train_cost, train_acc = net.SGD(
        training_data,
        epochs,
        mini_batch_size,
        eta,
        lmbda=lmbda,
        evaluation_data=validation_data,
        monitor_training_cost=True,
        monitor_training_accuracy=True,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True
    )

    # Final evaluation on test set
    test_accuracy = net.accuracy(test_data)
    print(f"Test accuracy: {test_accuracy} / {len(test_data)} = {test_accuracy/len(test_data):.2%}")

    # # Save model and metrics
    # net.save("mnist_network.json")
    # with open("training_metrics.json", "w") as f:
    #     json.dump({    
    #         "evaluation_cost": eval_cost,
    #         "evaluation_accuracy": eval_acc,
    #         "training_cost": train_cost,
    #         "training_accuracy": train_acc
    #     }, f)
    # print("Model and metrics saved.")


if __name__ == '__main__':
    main()
