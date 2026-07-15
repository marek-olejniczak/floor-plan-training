#!/usr/bin/env python3
import os
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
print("Testing polars import...")
import polars as pl
print("polars imported OK")
df = pl.DataFrame({"a": [1, 2, 3]})
print("polars basic ops OK")
print(df)
