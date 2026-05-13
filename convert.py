#!/usr/bin/env python
import math
import sys

import pandas as pd
import re
from debug import debug, set_debug_flag

# -----------------------------------------------------------------------------
in_file = "../reference.no_git/train-00000-of-00001.parquet"
train_file = 'data/gms8k/train.json'
validation_file = 'data/gms8k/validation.json'
enable_debug = False
split = True
max_rows = sys.maxsize
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

def write_dataframe(dataframe=None, file=None):
    # debug(f"first row: {dataframe.at[0, 'question']} => {dataframe.at[0, 'answer']}, {dataframe.at[0, 'expected_value']}")
    column_rename = {
        "question": "prompt",
        "answer": "response"
    }
    train_out = dataframe.rename(columns=column_rename).to_json(orient='records', index=False, indent=4)
    with open(file, "w") as outfile:
        outfile.write(train_out)

set_debug_flag(enable_debug)

dataframe = pd.read_parquet(in_file)
debug(f"columns = {dataframe.columns}")
# Dataframe
dataframe["expected_value"] = dataframe['answer'].str.extract(r'#### (\d+)')
dataframe["answer"] = dataframe['answer'].str.replace(r'(#### \d+)', '', regex=True,  flags=re.MULTILINE)

# Modify every answer to convert '###.*$' to another column - true_value
num_rows = min(max_rows, len(dataframe.index))
debug(f"num_rows = {num_rows}")
row_split = num_rows
if split:
    split_percent = 0.9
    row_split = math.trunc(num_rows * 0.9)

train_dataframe = dataframe[:row_split]
validation_dataframe = dataframe[row_split:]

write_dataframe(train_dataframe, train_file)
if split:
    write_dataframe(validation_dataframe, validation_file)
