import pytest
import tempfile
import os
from test_101.pytest_101 import add, train_toy_model


# Test if the add function works correctly
def test_add():
    assert add(1, 2) == 3


# Test if the add function raises an error when given a string
def test_add_logs_info(caplog):
    # caplog captures logs, here at INFO level
    add(2, 3)
    # Check that the correct log message appears
    assert "Adding 2 and 3, result is 6" in caplog.text


# Test if the model is trained under expected accuracy
def test_train_toy_model_accuracy():
    acc = train_toy_model()
    assert acc > 0.8  # Fail if accuracy is lower


# Test if the model is trained correctly with different sample sizes
@pytest.mark.parametrize("n_samples", [50, 100, 200])
def test_train_toy_model_accuracy_param(n_samples):
    acc = train_toy_model(n_samples=n_samples)
    assert acc > 0.8  # Accept a lower threshold for smaller datasets


# Test if the model is saved correctly
def test_train_toy_model_saves_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")
        acc = train_toy_model(n_samples=100, save_path=model_path)
        assert os.path.exists(model_path)
        assert os.path.getsize(model_path) > 0
        assert acc > 0.8
