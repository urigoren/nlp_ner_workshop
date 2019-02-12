# This code generates holdout data for the submission server
import os, json
ret = {}
for fname in os.listdir("test_lbl"):
    if not fname.endswith('.txt'):
        continue
    with open("test_lbl/" + fname, 'r') as f:
        ret[fname.replace(".txt", "")] = [l.strip() for l in f.readlines()]

with open('holdout.data', 'w') as f:
    json.dump(ret, f)
