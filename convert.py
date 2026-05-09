#!/usr/bin/env python
import pandas as pd
import re
from debug import debug, set_debug_flag

# -----------------------------------------------------------------------------
in_file = "../reference.no_git/train-00000-of-00001.parquet"
out_file = 'train.json'
enable_debug = False
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

set_debug_flag(enable_debug)

dataframe = pd.read_parquet(in_file)
debug(f"columns = {dataframe.columns}")
# Dataframe
dataframe["expected_value"] = dataframe['answer'].str.extract(r'#### (\d+)')
dataframe["answer"] = dataframe['answer'].str.replace(r'(#### \d+)', '', regex=True,  flags=re.MULTILINE)

# Modify every answer to convert '###.*$' to another column - true_value

# debug(f"first row: {dataframe.at[0, 'question']} => {dataframe.at[0, 'answer']}, {dataframe.at[0, 'expected_value']}")
column_rename = {
    "question": "prompt",
    "answer": "response"
}

json_out = dataframe.rename(columns=column_rename).to_json(orient='records', index=False, indent=4)

with open(out_file, "w") as outfile:
    outfile.write(json_out)
