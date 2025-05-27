from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import logging
import joblib


def add(a, b):
    result = a + b
    logging.warning(f"Adding {a} and {b}, result is {result}")
    return result


# simple ML example
def train_toy_model(n_samples=100, save_path=None):
    X, y = make_classification(n_samples=n_samples, n_features=4, random_state=42)
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    if save_path:
        joblib.dump(model, save_path)
    return acc
