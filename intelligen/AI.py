import numpy as np

# FUNCIONES DE ACTIVACIÓN

sigm = (
    lambda x: 1 / (1 + np.exp(-x)),
    lambda x: x* (1 - x)
)

    
# FUNCIONES DE ACTIVACIÓN

l2_cost = (
    lambda Yp, Yr: np.mean((Yp - Yr)**2),
    lambda Yp, Yr: (Yp-Yr)
)


class NeuralLayer:
    
    def __init__(self, n_conn, n_neur, act_f):
      
        self.act_f = act_f

        self.b = np.random.rand(n_neur)*2 -1
        self.W = np.random.rand(n_conn, n_neur)*2 -1
    
    def forward(self, prev):
        z = prev[1] @ self.W + self.b
        a = self.act_f[0](z)
        return z, a


class NeuralNet:

    def __init__(self, topology, act_f) -> None:
       self.layers = [NeuralLayer(topology[l], topology[l+1], act_f) for l, _ in enumerate(topology[:-1])]
    
    def __getitem__(self, i):
        return self.layers[i]
    
    def __iter__(self):
        yield from self.layers
    
    def __len__(self):
        return len(self.layers)
    
    def predict(self, X):
        out = [(None,X)]
        for layer in self:
            z, a = layer.forward(out[-1])
            out.append((z,a))
        
        return out

    def result(self, X):
        out = self.predict(X)
        return out[-1][1]

    def train(self, X, y, cost_f, lr=0.5):
        out = self.predict(X)
        deltas = []
        for l in reversed(range(0, len(self))):
            z = out[l+1][0]
            a = out[l+1][1]

            if l == len(self) - 1:
                # Calcular delta última capa
                deltas.insert(0, cost_f[1](a, y) * self[l].act_f[1](a))
            else:
                # Calcular delta respecto a capa previa
                deltas.insert(0, deltas[0] @ _W.T  * self[l].act_f[1](a))

            _W = self[l].W

            # Gradient descent
            self[l].b -= np.mean(deltas[0], axis=0)*lr # (n_neur,) - 
            self[l].W -= out[l][1].T @ deltas[0] * lr

        return out[-1][1]
