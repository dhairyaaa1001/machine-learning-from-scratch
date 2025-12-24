import numpy as np
import matplotlib.pyplot as plt
# --- 1️⃣ Create a small temporary dataset ---
# Let's say we have 6 tweets with binary sentiment: Positive=1, Negative=0
X_text = [
    "I love this product",      # Positive
    "This is amazing",          # Positive
    "Very happy with it",       # Positive
    "I hate this",              # Negative
    "This is terrible",         # Negative
    "Not good at all"           # Negative
]

y = np.array([1, 1, 1, 0, 0, 0])  # labels

# --- 2️⃣ Simple preprocessing ---
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c.isalpha() or c==' '])
    return text

X_clean = [clean_text(t) for t in X_text]

# --- 3️⃣ Convert text to simple features (Bag-of-Words) ---
# Very simple: count occurrence of each word
vocab = list(set(" ".join(X_clean).split()))
print("Vocabulary:", vocab)

def text_to_features(text):
    words = text.split()
    return [words.count(word) for word in vocab]

X = np.array([text_to_features(t) for t in X_clean])


class LogisticRegressionScratch:
    def __init__(self, lr=0.1, n_iter=100):
        self.losses = []
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = 0
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500) # WE CLIP THE INPUT SO THAT , NP.EXP(-Z) DOES NOT REACH VERY LARGE NUMBERS.
        return 1 / (1 + np.exp(-z))
    
    def binary_cross_entropy(self, y, y_hat):
        eps = 1e-9  # prevents log(0)
        return -np.mean(
            y * np.log(y_hat + eps) + #WHEN Y-HAT GETS 0 , LOG ZERO DOES NOT BREAK THE MODEL
            (1 - y) * np.log(1 - y_hat + eps)
        )

    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iter):
            y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
            error = y_pred - y
            self.weights -= self.lr * np.dot(X.T, error) / n_samples
            self.bias -= self.lr * np.sum(error) / n_samples
            loss = self.binary_cross_entropy(y, y_pred)
            self.losses.append(loss)
    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)



model = LogisticRegressionScratch(lr=0.1, n_iter=500)
model.fit(X, y)

preds = model.predict(X)
print("Predictions:", preds)
print("Actual     :", y)

accuracy = np.mean(preds == y)
print("Accuracy:", accuracy)


plt.plot(model.losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

