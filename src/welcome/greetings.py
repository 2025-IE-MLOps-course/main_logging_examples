"""
greetings.py — reusable greeting utilities

Provides small, testable helpers for generating greeting messages
"""


def format_greeting(name: str) -> str:
    """
    Return a friendly greeting for the given name

    Parameters
    ----------
    name : str
        Person to greet

    Returns
    -------
    str
        "Hello, <name>!"
    """
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    return f"Hello, {name}!"