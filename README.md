# CAP6938 Assignments

Coursework for CAP6938 at the University of Central Florida, by Longho Bernard Che.

| Folder | Assignment | Main deliverables |
|---|---|---|
| [`lab1/`](lab1/) | Dense knowledge representation: a CLIP-style dual encoder compared with an MLLM-based embedder on MMEB tasks | [notebook](lab1/dense_knowledge_representation_complete.ipynb), [report](lab1/dense_knowledge_representation_report.pdf) |
| [`lab2/`](lab2/) | Fast image search over ImageNet-1k: exact search compared with approximate indexing and quantization | [notebooks](lab2/submissions/), [report](lab2/report/report.pdf) |

Each lab folder has an `instructions.md` with the assignment text.

---

## Lab 2: Fast image search over ImageNet-1k

Given a query image, the system returns the 10 most similar images among the 1,281,167 images of
the ImageNet-1k training set. It measures what an approximate index (IVF) and quantization (SQ8,
PQ) gain and lose compared with exact search.

### Results

1,000 validation queries (one per class), measured on a 2-core Colab CPU, with IVF at nprobe = 16:

| Setting | Memory | Median latency | Precision@10 | Recall@10 |
|---|---|---|---|---|
| Flat (exact) | 3,936 MB | 366 ms | 0.701 | 1.000 |
| SQ8 (exact) | 984 MB | 216 ms | 0.701 | 0.981 |
| PQ64 (exact) | 83 MB | 46 ms | 0.606 | 0.329 |
| IVF-Flat | 3,959 MB | 3.5 ms | 0.701 | 0.975 |
| IVF-PQ64 | 106 MB | 1.5 ms | 0.682 | 0.492 |

- **IVF** is 105× faster than exact search, with the same precision.
- **IVF-PQ** is 241× faster and 37× smaller, and loses 0.02 precision.

The full analysis, including six failure cases, is in the [report](lab2/report/report.pdf).

### Contents

```text
lab2/
  instructions.md        assignment text
  submissions/           the three notebooks, executed on Colab with outputs
  results/               results.csv, summary.csv, failure_cases.csv and figures
  report/                report.tex and report.pdf (ACM two-column, 2 pages)
  tests/                 local test harness: synthetic dataset and headless notebook runner
```

### How to reproduce

The notebooks run on Google Colab and save everything to Google Drive under
`MyDrive/cap6938-hw2`. Run them in order:

| Notebook | Colab runtime | What it does | Time |
|---|---|---|---|
| [`01_embed`](lab2/submissions/01_embed.ipynb) | A100 GPU, High-RAM | Embeds all training images and 1,000 validation queries with CLIP ViT-L/14 | about 40 min |
| [`02_index_bench`](lab2/submissions/02_index_bench.ipynb) | Standard CPU | Builds the five indexes, measures latency, memory, precision and recall, and plots the trade-off | about 1–2 h |
| [`03_failures`](lab2/submissions/03_failures.ipynb) | Standard CPU | Selects failure cases and shows their images | a few minutes |

1. For a clean run, delete `MyDrive/cap6938-hw2` first. If a runtime disconnects, choose
   **Run all** again: finished work is skipped.
2. Every setting is read from a `LAB2_*` environment variable in the configuration cell of
   each notebook. The defaults are the values used for the results above: seed 42, IVF with 4,096
   lists, PQ with 64 sub-vectors, nprobe swept from 1 to 128, and so on. To change one, set
   `os.environ["LAB2_..."]` in a cell above the configuration cell.
3. Pinned packages are installed by the notebooks: `open_clip_torch==2.32.0` and
   `faiss-cpu==1.11.0`.
4. Optional: add an `HF_TOKEN` in Colab's **Secrets** panel for faster dataset downloads. Never
   put a token in a notebook or in this repository.

### Testing locally

`lab2/tests/` runs a notebook end to end on a small synthetic dataset that uses the real parquet
schema, using an untrained model on CPU. It checks that the pipeline runs: data handling, resume,
id mapping, checks and figures. It cannot reproduce real timings or accuracy.

```bash
python lab2/tests/make_fixture.py /tmp/lab2test/data
LAB2_ROOT=/tmp/lab2test/drive LAB2_LOCAL_DATA=/tmp/lab2test/data LAB2_ALLOW_CPU=1 \
LAB2_MODEL=ViT-B-32 LAB2_PRETRAINED= LAB2_NUM_WORKERS=0 LAB2_SHARDS=0-2 \
LAB2_NUM_TRAIN_FILES=3 LAB2_EXPECTED_TRAIN_ROWS=6000 LAB2_SMOKE_ROWS=256 LAB2_KEEP_PARQUET=1 \
  python lab2/tests/run_nb.py lab2/submissions/01_embed.ipynb /tmp/lab2test/out_01.ipynb <kernel>
```

`<kernel>` is the name of a Jupyter kernel with `torch`, `open_clip_torch==2.32.0`,
`faiss-cpu==1.11.0`, `pyarrow`, `pandas`, `matplotlib`, `nbclient` and `huggingface_hub`
installed. `run_nb.py` exits with code 1 if any cell fails.

### Building the report

```bash
cd lab2/report && pdflatex report.tex   # repeat until the log has no "Rerun" warning
```

---

## Lab 1: Dense knowledge representation

This lab compares a CLIP-style dual encoder with an MLLM-based embedder on queries from the four
MMEB categories: classification, retrieval, VQA and grounding. It analyses at least five failure
cases to explain which architectural property caused each difference. See
[`lab1/instructions.md`](lab1/instructions.md), the executed
[notebook](lab1/dense_knowledge_representation_complete.ipynb) and the
[report](lab1/dense_knowledge_representation_report.pdf).
