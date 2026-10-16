# Homework2 — Building a Fast Image Search System

---

## 1. The task

Build a search system over the ImageNet training set (about 1.3 million images) that takes a
query image and returns the 10 most similar images. Then measure what approximate indexing
and quantization buy you, and what they cost.

There are no fixed targets. Aim for a good balance of low latency, low memory, and high
accuracy, and explain the trade-offs you find.

---

## 2. Requirements

1. **Data.** The database is the ImageNet-1k training set at 256 px:
   [`benjamin-paine/imagenet-1k-256x256`](https://huggingface.co/datasets/benjamin-paine/imagenet-1k-256x256)
   (1,281,167 images in 40 parquet files, about 440 MB each, with the image and its class
   label).
2. **Embedding.** Embed the images with a multimodal embedding model of your choice (for
   example, a CLIP model).
3. **Approximate indexing.** Use at least one approximate nearest-neighbor index, such as
   IVF or HNSW.
4. **Quantization.** Use at least one quantization method, such as PQ, OPQ, or scalar
   quantization.
5. **Comparison.** Compare at least these four settings:

   | | no quantization | with quantization |
   | --- | --- | --- |
   | **exact search** (brute force) | baseline | |
   | **approximate index** | | |

   For the approximate settings, also vary the parameter that trades accuracy for speed
   (for example, nprobe or efSearch).
6. **Measurements.** For each setting, report at least:
   - **latency** per query: median and p99, one query at a time, on a Colab CPU runtime,
     excluding the time to embed the query
   - **memory** used by the index
   - **accuracy**: precision@10, the fraction of the top 10 results that have the same
     ImageNet class as the query

   Add anything else you think is worth profiling, such as embedding time, build time, or
   recall@10 against exact search.
7. **Qualitative analysis.** Show at least 5 failure cases, each with the query and its top
   results. For each, explain what went wrong: the embedding model, the approximate index,
   the quantization, or the class label itself (for example, a look-alike dog breed counted
   as wrong).
---

## 3. What to submit

1. **Report**, PDF, 1-2 pages.
2. **Notebook(s)**, `.ipynb` with outputs preserved.


## 4. Tools

Google Colab's free tier can be used. If you have access to other servers with available GPU resources, that's okay too.