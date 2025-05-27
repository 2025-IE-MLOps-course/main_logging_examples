import argparse
import mlflow
import sys
print("ARGV seen by Python:", sys.argv)


def main(args):
    print(f"Main received message: {args.message}")
    # Run the module as a separate MLflow project
    result = mlflow.run(
        uri="src/hello_module",
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
