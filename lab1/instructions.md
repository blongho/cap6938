** Topic 1, Dense Knowledge Representation**

 

---

## 1. The task

Compare two kinds of multimodal embedder on the same queries:

- a **CLIP-style dual encoder**
- an **MLLM-based embedder**

Pick **one or two MMEB datasets from each of the four categories**: image classification
(IC), image retrieval (IR), VQA, and grounding. Run both models over some queries from each,
and find **at least 5 failure cases** to analyze in depth.

You do not need to evaluate the full benchmark or reproduce leaderboard numbers. 

For each case, the central question is **why**. Not "the 2B model is better," but which
specific property of the architecture produced the difference.

---

## 2. Data

The data can be found on the Huggingface Hub at
**[`TIGER-Lab/MMEB-eval`](https://huggingface.co/datasets/TIGER-Lab/MMEB-eval)**.

You will need to figure out the dataset format and how to download (you only have to download what you need).

---

## 3. Models

Pick one CLIP-based and one MLLM-based embedding model. Two candidate models can be openai/clip-vit-base-patch32 (CLIP) and Qwen/Qwen3-VL-Embedding-2B (MLLM-based), but you can choose other models as well.

---

## 4. What to submit

1. **Report**, PDF, 1-2 pages. 
2. **Notebook**, `.ipynb` with outputs preserved.

## 5. Tool

You can use Google Colab's free-tier GPU for running the analysis.

