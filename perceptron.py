import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=20):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(input_size + 1)  # +1 para o bias
        self.history = []

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = np.dot(x, self.weights[1:]) + self.weights[0]
        return self.activation(z)

    def fit(self, X, y):
        for epoch in range(self.epochs):
            for xi, target in zip(X, y):
                z = np.dot(xi, self.weights[1:]) + self.weights[0]
                y_pred = self.activation(z)
                error = target - y_pred
                self.weights[1:] += self.lr * error * xi
                self.weights[0] += self.lr * error
            self.history.append(self.weights.copy())

    def plot_decision_boundary(self, X, y, title):
        for i, w in enumerate(self.history):
            plt.figure(figsize=(5, 4))
            plt.title(f"{title} - Época {i+1}")
            for j in range(len(X)):
                plt.scatter(X[j][0], X[j][1], c='blue' if y[j] == 0 else 'red', s=100, edgecolors='k')

            x_vals = np.array(plt.gca().get_xlim())
            if w[2] != 0:
                y_vals = -(w[1] * x_vals + w[0]) / w[2]
                plt.plot(x_vals, y_vals, '--k')
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

# Definindo a função de teste para as portas lógicas
def test_gate(gate):
    if gate == "AND":
        X = np.array([[0,0], [0,1], [1,0], [1,1]])
        y = np.array([0, 0, 0, 1])
    elif gate == "OR":
        X = np.array([[0,0], [0,1], [1,0], [1,1]])
        y = np.array([0, 1, 1, 1])
    elif gate == "XOR":
        X = np.array([[0,0], [0,1], [1,0], [1,1]])
        y = np.array([0, 1, 1, 0])
    else:
        raise ValueError("Função lógica não reconhecida.")

    print(f"\nTreinando Perceptron para função {gate}")
    perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
    perceptron.fit(X, y)
    perceptron.plot_decision_boundary(X, y, gate)

    print("Resultados da predição:")
    for xi in X:
        print(f"Entrada: {xi} -> Saída: {perceptron.predict(xi)}")

# Chamadas para testar AND, OR e XOR
test_gate("AND")
test_gate("OR")
test_gate("XOR")
