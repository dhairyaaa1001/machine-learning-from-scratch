import numpy as np

class LogisticRegressionScratch:
    """
    Binary Logistic Regression implemented from scratch using NumPy.
    Optimized using gradient descent with Binary Cross Entropy loss.
    """

    def __init__(self, lr=0.1, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.losses = []

    def _sigmoid(self, z):
        """
        Numerically stable sigmoid function.
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _binary_cross_entropy(self, y, y_hat):
        """
        Binary Cross Entropy loss.
        """
        eps = 1e-9
        return -np.mean(
            y * np.log(y_hat + eps) +
            (1 - y) * np.log(1 - y_hat + eps)
        )

    def fit(self, X, y):
        """
        Train the model using gradient descent.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iter):
            linear_output = np.dot(X, self.weights) + self.bias
            y_hat = self._sigmoid(linear_output)

            dw = np.dot(X.T, (y_hat - y)) / n_samples
            db = np.sum(y_hat - y) / n_samples

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            loss = self._binary_cross_entropy(y, y_hat)
            self.losses.append(loss)

    def predict_proba(self, X):
        """
        Predict probability estimates.
        """
        return self._sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X, threshold=0.5):
        """
        Predict binary class labels.
        """
        return (self.predict_proba(X) >= threshold).astype(int)
