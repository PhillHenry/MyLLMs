import random
import sys

import pandas as pd

def random_codes(codes: {str}) -> str:
    line = ""
    for i in range(int(random.random() * 20) + 1):
        line += f" {random.choice(codes)}"
    return line.strip()


def generate_icd10s(n: int, filename: str):
    codes = codes_from(filename)
    with open("/tmp/training_data.txt", "w") as f:
        for _ in range(n):
            f.write(f"{random_codes(codes)}\n")


def codes_from(filename):
    df = pd.read_excel(filename)
    codes = list(df["CODE"])
    codes = filter(lambda x: len(x.strip()) > 0, codes)
    codes = map(lambda x: x.strip(), codes)
    codes = list(codes)
    return codes


if __name__ == "__main__":
    generate_icd10s(100000, sys.argv[1])