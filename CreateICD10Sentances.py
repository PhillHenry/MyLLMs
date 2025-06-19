import random
import sys
from typing import Union

import numpy as np
import pandas as pd

from BertConfig import MY_CORPUS, MY_RESULTS
from BertUtils import is_diabetes

COMORBIDITIES = [
    "E6601", # obesity
    "H5450", # visual
    "H5451",
    "H54511A",
    "H54512A",
    "H5452",
    "H5452A1",
    "H5452A2",
    "H5460",
    "H5461",
    "H5462",
    "H547",
    "R0689", # vision
    "R069",
    "R0781",
    "T524X1A", # ketones
    "T524X1D",
    "T524X1S",
    "T524X2A",
    "T524X2D",
    "T524X2S",
    "T524X3A",
    "T524X3D",
    "T524X3S",
    "T524X4A",
    "T524X4D",
    "T524X4S",
]

def random_codes(codes: {str}) -> Union[str, bool]:
    line = ""
    diabetic = False
    for i in range(int(random.random() * 20) + 20):
        code = random.choice(codes)
        if is_diabetes(code) and not diabetic:
            diabetic = True
            N = max(1, len(COMORBIDITIES) // 2)  # Ensure we don't try to pick more than available
            k = random.randint(1, N)  # Random number of elements to pick
            random_comorbidities = set(random.sample(COMORBIDITIES, k))
            line = " ".join(random_comorbidities) + line
            line = f"{code} {line}"
        elif code in COMORBIDITIES and not diabetic:
            line = "E1010 " + line
            diabetic = True
        else:
            line += f" {code}"
    return line.strip(), diabetic


def generate_icd10s(n: int, filename: str):
    codes = codes_from(filename)
    diabetes = np.zeros(n)
    print(f"Writing {MY_CORPUS}...")
    with open(MY_CORPUS, "w") as f:
        for i in range(n):
            line, is_diabetes = random_codes(codes)
            if is_diabetes:
                diabetes[i] = 1
            f.write(f"{line}\n")
    diabetes = pd.DataFrame(diabetes)
    print(f"Writing {MY_RESULTS}...")
    diabetes.to_csv(MY_RESULTS, index=False, header=False)


def codes_from(filename: str):
    df = pd.read_excel(filename)
    codes = list(df["CODE"])
    codes = filter(lambda x: len(x.strip()) > 0, codes)
    codes = map(lambda x: x.strip(), codes)
    codes = list(codes)
    return codes


if __name__ == "__main__":
    n = 10_000
    print(f"""
    Outputs {n} rows of ICD-10 sentences and writes them to {MY_CORPUS}
    
    Expect one argument: full path to an Excel spreadsheet with ICD10 codes with the column "CODE".
    I got mine from https://www.cms.gov/files/document/valid-icd-10-list.xlsx
    """)
    generate_icd10s(n, sys.argv[1])