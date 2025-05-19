import random
import sys
from typing import Union

import numpy as np
import pandas as pd

from BertConfig import MY_CORPUS, MY_RESULTS


def random_codes(codes: {str}) -> Union[str, bool]:
    line = ""
    is_diabetes = False
    for i in range(int(random.random() * 20) + 20):
        code = random.choice(codes)
        line += f" {code}"
        if code.startswith("E0"):
            is_diabetes = True
    return line.strip(), is_diabetes


def generate_icd10s(n: int, filename: str):
    codes = codes_from(filename)
    diabetes = np.zeros(n)
    with open(MY_CORPUS, "w") as f:
        for i in range(n):
            line, is_diabetes = random_codes(codes)
            if is_diabetes:
                diabetes[i] = 1
            f.write(f"{line}\n")
    diabetes = pd.DataFrame(diabetes)
    diabetes.to_csv(MY_RESULTS, index=False, header=False)


def codes_from(filename: str):
    df = pd.read_excel(filename)
    codes = list(df["CODE"])
    codes = filter(lambda x: len(x.strip()) > 0, codes)
    codes = map(lambda x: x.strip(), codes)
    codes = list(codes)
    return codes


if __name__ == "__main__":
    n = 100000
    print(f"""
    Outputs {n} rows of ICD-10 sentences and writes them to {MY_CORPUS}
    
    Expect one argument: full path to an Excel spreadsheet with ICD10 codes with the column "CODE".
    I got mine from https://www.cms.gov/files/document/valid-icd-10-list.xlsx
    """)
    generate_icd10s(n, sys.argv[1])