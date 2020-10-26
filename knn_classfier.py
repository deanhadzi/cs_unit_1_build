from math import sqrt
import numpy as np


class KNearestNeighborsClassifier:
    """
    A simple attempt at creating a K-Nearest Neighbors algorithm.

    n_neighbors: int, default=5
        Number of neighbors to use by default in classification.
    """

    def __init__(self, n_neighbors=5):
        """Initialize the classifier."""
        self.neighbors = n_neighbors
        self.X = None
        self.y = None

    def fit(self, X_train, y_train):
        """
        Fit the train data. X_train can be multidimensional array.
        y_train can be one dimensional array that matches the length
        of the X_train.

        X: numpy array, training data
        y: numpy array, target values
        """
        self.X = X_train
        self.y = y_train

    def predict(self, X_test):
        """
        Predict the class labels for provided data.
        X_test: numpy array
        """
        predictions = []

        for row in X_test:
            prediction = self._make_prediction(
                row, self.X, self.y, self.neighbors)
            predictions.append(prediction)

        return np.array(predictions)

    def _eucl_dist(self, test_v, train_v):
        """
        Helper function to calculate the Euclidean distances of
        each test vector to each train vector.
        """

        dist = sum([(test_v[i] - train_v[i])**2 for i in range(len(test_v))])
        return sqrt(dist)

    def _get_neighbors(self, test_v, train_v, y_train, n_neighbors):
        """
        Helper function to calculate the nearest neighbors.
        """

        distances = []
        # Once the distance is calculated for each vector,
        # we attach the train vector, its associated y value
        # and actual distance to a list.
        for i in range(len(train_v)):
            dist = self._eucl_dist(test_v, train_v[i])
            distances.append((train_v[i], y_train[i], dist))
        # Sort the list based on the distance value.
        distances.sort(key=lambda item: item[2])

        # Get the number of neighbors from the distance list.
        # And return them.
        neighbors = []
        for i in range(n_neighbors):
            neighbors.append(distances[i])
        return neighbors

    def _make_prediction(self, test_v, train_v, y_train, n_neighbors):
        """
        Helper function to make prediction based on the number of neighbors.
        """

        neighbors = self._get_neighbors(
            test_v, train_v, y_train, n_neighbors
        )
        output_class = [row[-2] for row in neighbors]
        # Make the prediction based on the most voted class member.
        pred = max(set(output_class), key=output_class.count)
        return pred
