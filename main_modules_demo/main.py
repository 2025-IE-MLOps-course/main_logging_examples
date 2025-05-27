"""
main.py — Illustrates Python main script best practices

This script shows how to:
- Encapsulate execution logic inside a main() function
- Use argparse for flexible input handling
- Delegate tasks to modular functions
- Prevent top-level code execution
- Gracefully handle errors
"""

import argparse
from src.welcome.greetings import format_greeting


def main() -> None:
    """Script entry point"""
    parser = argparse.ArgumentParser(description="Greet the user by name")
    parser.add_argument("name", help="Name of the person to greet")
    args = parser.parse_args()

    try:
        print(format_greeting(args.name))
    except Exception as exc:
        print(f"Error: {exc}")
        raise


if __name__ == "__main__":
    main()
