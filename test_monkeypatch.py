# %%
import pytest
import time


def delay():

    time.sleep(15)  # Simulates a delay


def calculate_sum(a: int | float, b: int | float) -> str:

    delay()

    return f"Sum of the 2 Numbers is `{a + b}`"


# %%
calculate_sum(2, 2)
# %%


def test_calculate_sum(monkeypatch):

    def mock_delay():

        pass  # No delay

    monkeypatch.setattr("test_monkeypatch.delay", mock_delay)

    result = calculate_sum(2, 2)

    assert result == "Sum of the 2 Numbers is `4`"
