import numpy as np
import matplotlib.pyplot as plt

from linear_models.logistic_regression import LogisticRegressionScratch
from datasets.toy_text_data import X_text, y
from utils.text_vectorizer import clean_text, build_vocab, text_to_features


# Preprocess text
X_clean = [clean_text(t) for t in X_text]
vocab = build_vocab(X_clean)
X = np.array([text_to_features(t, vocab) for t in X_clean])
y = np.array(y)

# Train model
model = LogisticRegressionScratch(lr=0.1, n_iter=500)
model.fit(X, y)

# Predictions
preds = model.predict(X)
accuracy = np.mean(preds == y)

print("Predictions:", preds)
print("Actual     :", y)
print("Accuracy   :", accuracy)

# Plot loss
plt.plot(model.losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
