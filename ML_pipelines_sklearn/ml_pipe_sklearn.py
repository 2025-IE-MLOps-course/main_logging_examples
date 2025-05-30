# %% Import libraries ----------------------
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# %% Load dataset ----------------------
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.head()

# %% Typical data splitting ----------------------

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

# %% ----------------------
y_train.value_counts(), y_test.value_counts()

# full set
df['target'].value_counts()

# %% Manual preprocessing ----------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled[:5]

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
sum_features.fit_transform(X_train).head()
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

# %% Enforcing leakage-proof pipeline ----------------------

# Define the preprocessing pipeline (SumFeatures + Scaler only)
preprocessing_pipeline = Pipeline([
    ('sum_features', SumFeatures()),
    ('scaler', StandardScaler())
])

# Fit only on training data
preprocessing_pipeline.fit(X_train)

# Save the fitted preprocessing pipeline
with open('preprocessing_pipeline.pkl', 'wb') as f:
    pickle.dump(preprocessing_pipeline, f)

# %% Separate model training from preprocessing ----------------------

# Transform data for model training
X_train_processed = preprocessing_pipeline.transform(X_train)
X_test_processed = preprocessing_pipeline.transform(X_test)

# %% Train the model on processed data
# Save the trained model (on processed data)
model_separated = LogisticRegression()
model_separated.fit(X_train_processed, y_train)

with open('model_separated.pkl', 'wb') as f:
    pickle.dump(model_separated, f)

print("Saved preprocessing pipeline and model separately.")


# %% Load and apply artifacts for prediction ----------------------

with open('preprocessing_pipeline.pkl', 'rb') as f:
    loaded_preprocessor = pickle.load(f)

with open('model_separated.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Prepare a new sample (or batch)
new_sample = X.iloc[[100]]

# Apply preprocessing
sample_processed = loaded_preprocessor.transform(new_sample)

# Predict with the model
pred = loaded_model.predict(sample_processed)
print(f"Prediction from separated artifacts: {pred[0]}")

# %%
