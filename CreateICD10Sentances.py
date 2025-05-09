import random
import sys

import pandas as pd

OUTPUT_FILE = "/tmp/training_data.txt"


def random_codes(codes: {str}) -> str:
    line = ""
    for i in range(int(random.random() * 20) + 20):
        line += f" {random.choice(codes)}"
    return line.strip()


def generate_icd10s(n: int, filename: str):
    codes = codes_from(filename)
    with open(OUTPUT_FILE, "w") as f:
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
    n = 100000
    print(f"""
    Outputs {n} rows of ICD-10 sentences and writes them to {OUTPUT_FILE}
    
    Expect one argument: full path to an Excel spreadsheet with ICD10 codes with the column "CODE".
    I got mine from https://www.cms.gov/files/document/valid-icd-10-list.xlsx
    """)
    generate_icd10s(n, sys.argv[1])