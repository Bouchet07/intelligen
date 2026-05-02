import pytest
import numpy as np
from intelligen.AI import NeuralNet, sigm

def test_neural_net_instantiation():
    topology = [2, 3, 1]
    net = NeuralNet(topology, sigm)
    assert len(net) == 2
    assert net[0].W.shape == (2, 3)
    assert net[1].W.shape == (3, 1)

def test_neural_net_predict():
    topology = [2, 3, 1]
    net = NeuralNet(topology, sigm)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predictions = net.result(X)
    assert predictions.shape == (4, 1)
