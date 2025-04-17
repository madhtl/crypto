import os
import pandas as pd
import argparse


def generate_dataset(name: str, input_files: list[str]):
    output_file = pd.DataFrame()
    for i, file_path in enumerate(input_files):
        if not file_path.startswith("crypto/coin_"):
            file_path = os.path.join("crypto", f"coin_{file_path}")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        file = pd.read_csv(file_path)
        if 'Low' not in file.columns:
            raise ValueError(f"'Low' column not found in {file_path}")
        col_name = f"Low: {os.path.basename(file_path)[:-4]}"
        output_file[col_name] = file['Low'].reset_index(drop=True)
    print(file.head())
    print(file['Low'].head())
    output_file.to_csv(name)
    return output_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a Low dataset from cryptocurrency data files.")
    parser.add_argument("-o", required=True)  # flag arg
    parser.add_argument("input_files", nargs="+")  # positional arg
    args = parser.parse_args()
    print(args)
    transform = generate_dataset(args.o, args.input_files)
    print(transform.head())
