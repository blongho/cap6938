# CAP6938 Assignments

Coursework for CAP6938 at the University of Central Florida, by Longho Bernard Che.

| Folder | Assignment | Main deliverables |
|---|---|---|
| [`lab1/`](lab1/) | Dense knowledge representation: a CLIP-style dual encoder compared with an MLLM-based embedder on MMEB tasks | [notebook](lab1/dense_knowledge_representation_complete.ipynb), [report](lab1/dense_knowledge_representation_report.pdf) |
| [`lab2/`](lab2/) | Fast image search over ImageNet-1k: exact search compared with approximate indexing and quantization | [notebooks](lab2/submissions/), [report](lab2/report/report.pdf) |

Each lab folder has an `instructions.md` with the assignment text.

---

## Lab 2: Fast image search over ImageNet-1k

**Live demo: https://blongho-imagenet-image-search.hf.space**

A CLIP ViT-L/14 + FAISS image search over the 1,281,167 ImageNet-1k training images. It compares exact search
with approximate indexing (IVF) and quantization (SQ8, PQ). IVF is 105× faster than exact search with the
same precision. IVF-PQ is 241× faster and 37× smaller, for 0.02 lower precision.

Results, how to reproduce, the live demo and its deployment, and local tests are in
[`lab2/README.md`](lab2/README.md).

---

## Lab 1: Dense knowledge representation

This lab compares a CLIP-style dual encoder with an MLLM-based embedder on queries from the four
MMEB categories: classification, retrieval, VQA and grounding. It analyses at least five failure
cases to explain which architectural property caused each difference. See
[`lab1/instructions.md`](lab1/instructions.md), the executed
[notebook](lab1/dense_knowledge_representation_complete.ipynb) and the
[report](lab1/dense_knowledge_representation_report.pdf).
