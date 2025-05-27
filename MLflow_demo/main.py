import argparse
import mlflow
import sys
import os


def main(args):

    # Get the absolute path to the directory where main.py is located
    main_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the submodule's URI
    module_uri = os.path.join(main_script_dir, "src", "hello_module")

    print(f"Calling submodule with absolute URI: {module_uri}")

    # Run the module as a separate MLflow project using the absolute path
    result = mlflow.run(
        uri=module_uri,
        entry_point="main",
        parameters={"message": args.message},
        env_manager="conda"
    )
    print(f"Module run finished: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", type=str, default="Hello from Main!")
    args = parser.parse_args()
    main(args)
