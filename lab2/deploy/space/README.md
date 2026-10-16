---
title: ImageNet Image Search
emoji: 🐠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: CLIP ViT-L/14 + FAISS IVF-PQ search over ImageNet-1k
---

# ImageNet-1k image search

Upload an image, or pick a random ImageNet validation image, and the app returns the 10 most similar of
the 1,281,167 ImageNet-1k training images.

**How it works**

1. The query image is embedded with OpenAI CLIP ViT-L/14 (768 dimensions, unit length). This is the same model and
   preprocessing that embedded the database.
2. A FAISS IVF-PQ index (4,096 clusters, 64-byte product-quantized codes, about 106 MB) finds the nearest
   neighbours by cosine similarity. It searches the `nprobe` clusters closest to the query.
3. The 10 result images are streamed from the
   [`benjamin-paine/imagenet-1k-256x256`](https://huggingface.co/datasets/benjamin-paine/imagenet-1k-256x256) parquet
   files, one row group (about 100 images) at a time. No ImageNet images are stored in this Space.

**Hardware:** ZeroGPU. Only the CLIP embedding of the query runs on the GPU (marked `@spaces.GPU`); the FAISS
search and image fetching run on the CPU. Each search uses a few seconds of the visitor's daily ZeroGPU quota
(5 minutes for free accounts). The app also runs unchanged on CPU hardware.

**Measured on 1,000 validation queries (2-core CPU, nprobe 16):** precision@10 0.682, recall@10 0.492 against exact
search, 1.5 ms median search time. This is 241× faster than exact search, with an index 37× smaller.

This Space was built for CAP6938 Homework 2 (University of Central Florida) by Longho Bernard Che. ImageNet images
may be used for non-commercial research and education only.

## Configuration

Set these in the Space's **Settings → Variables and secrets**.

| Name | Kind | Default | Purpose |
|---|---|---|---|
| `ASSETS_REPO` | variable | (required) | Dataset repository holding `ivfpq.faissindex`, `db_labels.npy`, `assets.json`, `class_names.json` |
| `HF_TOKEN` | **secret** | none | Read token; needed if `ASSETS_REPO` is private |
| `TOPK` | variable | `10` | Number of results |
| `NUM_THREADS` | variable | all cores | CPU threads for CLIP and FAISS |
| `FETCH_WORKERS` | variable | `10` | Parallel image downloads |
| `ROWGROUP_CACHE` | variable | `64` | Parquet row groups (about 100 images each) kept in memory |
