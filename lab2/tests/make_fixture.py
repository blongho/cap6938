"""Build a small synthetic stand-in for benjamin-paine/imagenet-1k-256x256.

Same layout as the real dataset: data/train-0000N-of-00003.parquet (3 x 2,000 rows) and
data/validation-0000N-of-00002.parquet (2 x 300 rows); an `image` struct<bytes, path> column,
an int `label` column, and Hugging Face schema metadata whose label names end with a 'none'
placeholder, like the real 1,001-name list. Images are 64 px JPEGs coloured by class.

Usage: python make_fixture.py <out_dir>
"""
import io
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

out = sys.argv[1]
os.makedirs(f"{out}/data", exist_ok=True)
NC = 10
names = [f"class{c}, synonym{c}" for c in range(NC)] + ["none"]
meta = {b"huggingface": json.dumps({"info": {"features": {
    "image": {"_type": "Image"}, "label": {"names": names, "_type": "ClassLabel"}}}}).encode()}
rng = np.random.default_rng(0)
base = rng.integers(0, 255, size=(NC, 3))


def write(path, n):
    labels = rng.integers(0, NC, n)
    blobs = []
    for c in labels:
        arr = np.clip(base[c] + rng.normal(0, 40, (64, 64, 3)), 0, 255).astype(np.uint8)
        b = io.BytesIO()
        Image.fromarray(arr).save(b, "JPEG")
        blobs.append(b.getvalue())
    img = pa.StructArray.from_arrays([pa.array(blobs, pa.binary()), pa.array([None] * n, pa.string())],
                                     ["bytes", "path"])
    t = pa.table({"image": img, "label": pa.array(labels, pa.int64())}).replace_schema_metadata(meta)
    pq.write_table(t, path, row_group_size=500)


for i in range(3):
    write(f"{out}/data/train-{i:05d}-of-00003.parquet", 2000)
for i in range(2):
    write(f"{out}/data/validation-{i:05d}-of-00002.parquet", 300)
print("fixture written to", out)
