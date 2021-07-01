import numpy as np
from sklearn.model_selection import train_test_split

# Local modules
from perceptron import Perceptron


class ModelBuilder:
    def __init__(self, data, feature_indexes, label_values):
        self.data = data
        self.feature_indexes = feature_indexes
        self.label_values = label_values

        self.feature_names = None
        self.label_name = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.weights = None
        self.bias = None

        self.train_accuracy = None
        self.test_accuracy = None

    # Function to calculate the accuracy (ranging from 0 to 1), given a collection of actual values and a collection
    # of predicted values.
    @staticmethod
    def accuracy(y_true, y_pred):
        # Calculate the accuracy number of matches between the actual and predicted values divided by the total.
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    # Function to build the perceptron model from the features corresponding to feature_indexes and for feature samples
    # with output values corresponding to label_values (e.g. Tyrannosaurus and Stegossaurus).
    def build(self):
        print('Selected features:')
        print(self.feature_indexes)

        # Input names from header row.
        self.feature_names = self.data[0, self.feature_indexes]

        # Output name header row.
        self.label_name = self.data[0, -1]

        print('Feature names:')
        print(self.feature_names)

        print('Label name:')
        print(self.label_name)

        print('Label values to build model for:')
        print(self.label_values)

        # Remove header from data.
        data = self.data[1:]

        # Filter data to keep samples for specific labels (e.g. Tyrannosaurus and Stegossaurus).
        dinosaurs = np.array([row for row in data if row[-1] in self.label_values])

        # Convert feature values to float.
        X = dinosaurs[:, self.feature_indexes].astype(float)

        print('First sample (X[0]):')
        print(X[0])

        # Set y to values of last column. This will only contain values specified in label_values
        # (e.g. Tyrannosaurus and Stegossaurus).
        y = dinosaurs[:, -1]

        # Convert label values to corresponding indexes of values in label_values
        # (e.g. Tyrannosaurus => 0, Stegossaurus => 1)
        # The corresponding index is determined by calling the index method of the label_values list, which will return
        # the index of the first occurance of y_value.
        y = np.array([self.label_values.index(y_value) for y_value in y])

        print('Converted label values (y):')
        print(y)

        # Split the data into a training set containing 80% of the samples/outputs and a test set containing 20% of the
        # samples/outputs.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=123)

        # Create an instance of the Perceptron class, specifying the alpha (learning rate) and the number of iterations.
        p = Perceptron(alpha=0.01, iterations=100)

        # Fit the training data to the model.
        p.fit(self.X_train, self.y_train, init_weight_value=0, init_bias=0)

        self.weights = p.weights
        self.bias = p.bias

        # Predict the output values for the training set to determine how well the model fits the training set.
        predictions_train = p.predict(self.X_train)

        # Calculate the accuracy of the predicted values for the training set. this value indicates how well the model
        # fits the training set.
        self.train_accuracy = ModelBuilder.accuracy(self.y_train, predictions_train)
        print("Perceptron classification training-set accuracy", self.train_accuracy)

        # Predict the output values for the test set.
        predictions_test = p.predict(self.X_test)

        print('actual test-set y values:')
        print(self.y_test)

        print('predicted test-set y values:')
        print(predictions_test)

        # Calculate the accuracy of the predicted values for the test set to determine how well the model performs with
        # unseen data.
        self.test_accuracy = ModelBuilder.accuracy(self.y_test, predictions_test)
        print("Perceptron classification test-set accuracy", self.test_accuracy)
