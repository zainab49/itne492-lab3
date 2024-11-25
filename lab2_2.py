import numpy as np  # Import numpy for array operations

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

# Mean Squared Error loss function
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Neural Network class
class OurNeuralNetwork:
    """
    A neural network with:
        - 2 inputs
        - A hidden layer with 2 neurons (h1, h2)
        - An output layer with 1 neuron (o1)
    """
    def __init__(self):
        # Initialize weights and biases with random values
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        # x is a numpy array with 2 elements
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        """
        - data is a (n x 2) numpy array, n = # of samples in the dataset
        - all_y_trues is a numpy array with n elements
        """
        learn_rate = 0.1
        epochs = 1000  # Number of training iterations

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Feedforward step
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # --- Calculate partial derivatives
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # --- Update weights and biases
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            # --- Calculate loss every 10 epochs
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print(f"Epoch {epoch} loss: {loss:.3f}")

# Extended dataset
data = np.array([
    [-2, -1],   # Alice
    [25, 6],    # Bob
    [17, 4],    # Charlie
    [-15, -6],  # Diana
    [8, 3],     # New person
    [12, 4],    # New person
    [-10, -4],  # New person
    [20, 5],    # New person
    [-5, -2],   # New person
    [15, 5],    # New person
    [18, 6],    # New person
    [-12, -5],  # New person
    [10, 3],    # New person
    [-8, -3],   # New person
    [25, 8],    # New person
    [22, 7],    # New person
    [-20, -8],  # New person
    [5, 2],     # New person
    [-25, -9],  # New person
    [30, 9],    # New person
    [35, 10],   # New person
    [-18, -7],  # New person
    [13, 4],    # New person
    [-7, -3],   # New person
])

all_y_trues = np.array([
    1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1,
])

# Train the neural network
network = OurNeuralNetwork()
network.train(data, all_y_trues)

# Test predictions
test_data = np.array([
    [-6, -2], [22, 7], [10, 3], [-18, -8], [15, 5],
    [8, 2], [-12, -4], [30, 10], [-5, -1], [20, 6],
])

expected_labels = np.array([1, 0, 0, 1, 0, 0, 1, 0, 1, 0])  # True labels
predictions = np.array([1 if network.feedforward(x) > 0.5 else 0 for x in test_data])

# Calculate success rate
correct_predictions = np.sum(predictions == expected_labels)
success_rate = (correct_predictions / len(expected_labels)) * 100

print("Predictions:", predictions)
print("Expected:", expected_labels)
print(f"Success Rate: {success_rate:.2f}%")
