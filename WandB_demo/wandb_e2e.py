# %% Install and import libraries
import wandb
import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import pickle
import os

# %% Initialize W&B project -----------------------
wandb.login()
wandb.init(project="wandb_e2e", name="logistic_regression")

# %% Load and split dataset -----------------------
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
labels = data.target_names


X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)

# %% Define model parameters and configuration -----------------------
wandb.config = {
    "model_type": "LogisticRegression",
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 100
}

# %% Create and train pipeline -----------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        penalty=wandb.config['penalty'],
        C=wandb.config['C'],
        solver=wandb.config['solver'],
        max_iter=wandb.config['max_iter']

    ))
])

pipeline.fit(X_train, y_train)

# %% Evaluate model and log metrics -----------------------
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
f1 = f1_score(y_test, y_pred, average='weighted')

wandb.log({
    "accuracy": accuracy,
    "roc_auc": roc_auc,
    "f1_score": f1
})

# %% Log model artifact and save model -----------------------
model_filename = "logistic_regression_model.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(pipeline, f)

artifact = wandb.Artifact(name="logistic_regression_model", type="model")
artifact.add_file(model_filename)
wandb.log_artifact(artifact)

# %% Visualize classifier performance -----------------------
wandb.sklearn.plot_classifier(
    pipeline.named_steps['classifier'],
    X_train,
    X_test,
    y_train,
    y_test,
    y_pred,
    y_proba,
    labels,
    model_name="Logistic Regression"
)

# %% Compare with Decision Tree Classifier -----------------------
wandb.init(project="wandb_e2e", name="decision_tree")

wandb.config = {
    "model_type": "DecisionTreeClassifier",
    "criterion": "gini",
    "max_depth": 3
}

pipeline_dt = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', DecisionTreeClassifier(
        criterion=wandb.config['criterion'],
        max_depth=wandb.config['max_depth']
    ))
])

pipeline_dt.fit(X_train, y_train)

y_pred_dt = pipeline_dt.predict(X_test)
y_proba_dt = pipeline_dt.predict_proba(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
roc_auc_dt = roc_auc_score(y_test, y_proba_dt, multi_class='ovo')
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')

wandb.log({
    "accuracy": accuracy_dt,
    "roc_auc": roc_auc_dt,
    "f1_score": f1_dt
})

model_filename_dt = "decision_tree_model.pkl"
with open(model_filename_dt, "wb") as f:
    pickle.dump(pipeline_dt, f)

artifact_dt = wandb.Artifact(name="decision_tree_model", type="model")
artifact_dt.add_file(model_filename_dt)
wandb.log_artifact(artifact_dt)

wandb.sklearn.plot_classifier(
    pipeline_dt.named_steps['classifier'],
    X_train,
    X_test,
    y_train,
    y_test,
    y_pred_dt,
    y_proba_dt,
    labels,
    model_name="Decision Tree"
)

# %% Set up and run a sweep -----------------------
sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'max_depth': {
            'values': [2, 3, 4, 5]
        },
        'criterion': {
            'values': ['gini', 'entropy']
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="wandb_e2e")


def train():
    wandb.init()
    config = wandb.config

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(
            criterion=config['criterion'],
            max_depth=config['max_depth']
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    f1 = f1_score(y_test, y_pred, average='weighted')

    wandb.log({
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "f1_score": f1
    })


wandb.agent(sweep_id, function=train, count=5)

# %% Finish W&B run -----------------------
wandb.finish()

# %%
