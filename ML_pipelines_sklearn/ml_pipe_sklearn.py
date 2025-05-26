# %% Import libraries ----------------------
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# %% Load dataset ----------------------
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.head())

# %% Typical data splitting ----------------------

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

# %% Manual preprocessing ----------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled[:5])

# %% Manual model fitting ----------------------

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# %% Manual model evaluation ----------------------

y_pred = model.predict(X_test_scaled)
accuracy = model.score(X_test_scaled, y_test)
print(f"Test accuracy: {accuracy:.2f}")

# %% Basic pipeline w/ scaler and classifier ----------------------

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
# Fit the pipeline
pipe.fit(X_train, y_train)
accuracy = pipe.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

# %% Simple Custom Transformer ----------------------


class SumFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['feature_sum'] = X_.sum(axis=1)
        return X_


# Test custom transformer
sum_features = SumFeatures()
print(sum_features.fit_transform(X_train).head())
# %% Pipeline with custom transformer ----------------------

pipe_with_custom = Pipeline([
    ('sum_features', SumFeatures()),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

pipe_with_custom.fit(X_train, y_train)
accuracy = pipe_with_custom.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

# %% Predictions with custom transformer ----------------------

sample = X.iloc[[100]]
pred = pipe_with_custom.predict(sample)
print(f"Prediction for sample: {pred[0]}")
