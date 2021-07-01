import numpy as np


class Perceptron:

    def __init__(self, alpha=0.01, iterations=100):
        self.alpha = alpha
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y, init_weight_value=0, init_bias=0):
        n_samples, n_features = X.shape

        # Initialise weights vector and bias.
        self.weights = np.ones(n_features) * init_weight_value
        self.bias = init_bias

        print('alpha:', self.alpha)
        print('iterations:', self.iterations)

        print('initial weights:')
        print(self.weights)

        print('initial bias:')
        print(self.bias)

        for _ in range(0, self.iterations):
            for x_row_index, x_row in enumerate(X):
                # Predict y for this x_row using current weights and bias.
                y_predicted = self.predict(x_row)
                
                # Determine update value for this iteration.
                update = self.alpha * (y[x_row_index] - y_predicted)

                # Update weights.
                self.weights += update * x_row

                # Update bias.
                self.bias += update

        print('final weights:')
        print(self.weights)

        print('final bias:')
        print(self.bias)

    def predict(self, X):
        # Calculate linear output.
        linear_output = np.dot(X, self.weights) + self.bias

        # Return predicted y.
        return Perceptron.activation_function(linear_output)

    # Unit step activation function.
    @staticmethod
    def activation_function(x):
        return np.where(x >= 0, 1, 0)
        

