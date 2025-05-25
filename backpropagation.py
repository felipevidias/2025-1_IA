import numpy as np
import matplotlib.pyplot as plt

# Funções de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - x**2

# Classe da rede neural
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, learning_rate=0.1, activation='sigmoid', use_bias=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 1
        self.lr = learning_rate
        self.use_bias = use_bias

        # Pesos da entrada para a camada oculta
        self.w1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.b1 = np.random.uniform(-1, 1, (1, hidden_size)) if use_bias else np.zeros((1, hidden_size))

        # Pesos da camada oculta para a saída
        self.w2 = np.random.uniform(-1, 1, (hidden_size, self.output_size))
        self.b2 = np.random.uniform(-1, 1, (1, self.output_size)) if use_bias else np.zeros((1, self.output_size))

        # Função de ativação
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_deriv = sigmoid_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        else:
            raise ValueError("Função de ativação não suportada.")

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            # Forward pass
            z1 = np.dot(X, self.w1) + self.b1
            a1 = self.activation(z1)

            z2 = np.dot(a1, self.w2) + self.b2
            a2 = self.activation(z2)

            # Backward pass
            error = y - a2
            d_output = error * self.activation_deriv(a2)

            error_hidden = d_output.dot(self.w2.T)
            d_hidden = error_hidden * self.activation_deriv(a1)

            # Atualização dos pesos
            self.w2 += a1.T.dot(d_output) * self.lr
            self.w1 += X.T.dot(d_hidden) * self.lr

            if self.use_bias:
                self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.lr
                self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.lr

            # Log de erro (opcional)
            if epoch % 1000 == 0:
                loss = np.mean(np.abs(error))
                print(f"Época {epoch}: Erro médio = {loss:.4f}")

    def predict(self, X):
        z1 = np.dot(X, self.w1) + self.b1
        a1 = self.activation(z1)

        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self.activation(z2)
        return np.round(a2)

# Funções booleanas
def get_dataset(gate):
    if gate == "AND":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [0], [0], [1]])
    elif gate == "OR":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [1]])
    elif gate == "XOR":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
    else:
        raise ValueError("Função lógica inválida.")
    return X, y

# Função para testar um gate
def run_experiment(gate="XOR", learning_rate=0.1, activation='sigmoid', use_bias=True):
    print(f"\n----- Testando {gate} com taxa {learning_rate}, ativação {activation}, bias {use_bias} -----")
    X, y = get_dataset(gate)

    # Criar rede com 2 neurônios na entrada e 2 na camada oculta
    nn = NeuralNetwork(input_size=2, hidden_size=2, learning_rate=learning_rate, activation=activation, use_bias=use_bias)
    nn.train(X, y, epochs=10000)

    predictions = nn.predict(X)
    print("Entradas:", X)
    print("Saídas esperadas:", y.ravel())
    print("Saídas da rede:", predictions.ravel())
    print("Acertos:", np.sum(predictions == y), "/", len(y))

# Executar experimentos
run_experiment("AND", learning_rate=0.1, activation="sigmoid", use_bias=True)
run_experiment("OR", learning_rate=0.1, activation="tanh", use_bias=True)
run_experiment("XOR", learning_rate=0.1, activation="sigmoid", use_bias=True)
