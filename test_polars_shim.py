#!/usr/bin/env python3
"""Test polars replacement."""
import sys
sys.path.insert(0, "/home/marek_olejniczak/projects/trening/.venv/lib/python3.11/site-packages")

import polars as pl
print("=== import OK ===")

# Test read_csv
import io
csv_data = "a,b,c\n1,2,3\n4,5,6"
df = pl.read_csv(io.StringIO(csv_data))
print("read_csv:", df.to_dict(as_series=False))

# Test DataFrame
df2 = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
print("DataFrame:", df2.to_dict())

# Test write_csv
s = df2.write_csv()
print("write_csv:", repr(s))

print("=== ALL TESTS PASSED ===")
