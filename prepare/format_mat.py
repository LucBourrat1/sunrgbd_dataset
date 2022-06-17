import json
import argparse
import os
from scipy import io

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpath", type=str, required=True)
    parser.add_argument("--foutput", type=str, default=".")
    return parser

def main():
    args = parser().parse_args()

    json_file = io.loadmat(args.fpath)

    with open(args.foutput, "w") as f:
        json.dump(json_file, f, indent=4)

    return


if __name__ == "__main__":
    main()
