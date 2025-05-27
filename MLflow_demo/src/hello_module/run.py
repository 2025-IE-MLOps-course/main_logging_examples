import argparse


def main(args):
    print(f"Hello Module received message: {args.message}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", type=str,
                        default="Hello from hello_module")
    args = parser.parse_args()
    main(args)
