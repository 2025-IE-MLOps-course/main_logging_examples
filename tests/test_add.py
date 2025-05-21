import pytest
from test_101.pytest_101 import add, train_toy_model


def test_add():
    assert add(1, 2) == 3


def test_add_logs_info(caplog):
    # caplog captures logs, here at INFO level
    add(2, 3)
    # Check that the correct log message appears
    assert "Adding 2 and 3, result is 5" in caplog.text


def test_train_toy_model_accuracy():
    acc = train_toy_model()
    assert acc > 0.8  # Fail if accuracy is lower


@pytest.mark.parametrize("n_samples", [50, 100, 200])
def test_train_toy_model_accuracy_param(n_samples):
    acc = train_toy_model(n_samples=n_samples)
    assert acc > 0.8  # Accept a lower threshold for smaller datasets