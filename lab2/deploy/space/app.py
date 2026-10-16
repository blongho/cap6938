"""ImageNet-1k image search: CLIP ViT-L/14 embeddings + FAISS IVF-PQ, results streamed from the dataset.

Configuration (environment variables, set as Space variables or secrets):
  ASSETS_REPO        Hugging Face dataset repo holding the index and metadata (required unless ASSETS_DIR)
  ASSETS_DIR         local folder with the same files, used instead of ASSETS_REPO (for testing)
  HF_TOKEN           read token, needed when ASSETS_REPO is private (Space secret)
  LOCAL_DATA         folder with local copies of the dataset parquet files (testing); default: stream from the Hub
  TOPK               number of results (default 10)
  NUM_THREADS        CPU threads for CLIP (on CPU hardware) and FAISS (default: all cores)
  FETCH_WORKERS      parallel image downloads (default 10)
  ROWGROUP_CACHE     parquet row groups kept in memory (default 64)
  SOURCE_URL         link to the source code shown at the bottom of the app
"""
import io
import json
import os
import threading
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor

# On ZeroGPU hardware, `spaces` must be imported before torch. Outside Spaces it may be missing, and then
# the decorator below does nothing.
try:
    import spaces
    gpu = spaces.GPU
except ImportError:
    def gpu(fn=None, **_):
        return fn if fn is not None else (lambda f: f)

# torch must be imported before faiss: both bundle an OpenMP runtime, and loading faiss's first makes
# torch crash when it starts its own threads (seen on macOS; harmless elsewhere).
import torch
import open_clip
import faiss
import gradio as gr
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download
from PIL import Image

ASSETS_REPO = os.environ.get("ASSETS_REPO", "")
ASSETS_DIR = os.environ.get("ASSETS_DIR", "")
LOCAL_DATA = os.environ.get("LOCAL_DATA", "")
TOPK = int(os.environ.get("TOPK", 10))
NUM_THREADS = int(os.environ.get("NUM_THREADS", os.cpu_count() or 1))
FETCH_WORKERS = int(os.environ.get("FETCH_WORKERS", 10))
ROWGROUP_CACHE = int(os.environ.get("ROWGROUP_CACHE", 64))
SOURCE_URL = os.environ.get("SOURCE_URL", "https://github.com/blongho/cap6938/tree/master/lab2")
NPROBE_CHOICES = [1, 2, 4, 8, 16, 32, 64, 128]
ASSET_FILES = ["assets.json", "class_names.json", "db_labels.npy", "ivfpq.faissindex"]


def asset_path(name):
    if ASSETS_DIR:
        return os.path.join(ASSETS_DIR, name)
    if not ASSETS_REPO:
        raise RuntimeError("Set ASSETS_REPO (Space variable) or ASSETS_DIR to the folder with the index files.")
    return hf_hub_download(ASSETS_REPO, name, repo_type="dataset")


# --- Assets: index, labels, metadata ---------------------------------------------------------------
paths = {name: asset_path(name) for name in ASSET_FILES}
with open(paths["assets.json"]) as f:
    ASSETS = json.load(f)
with open(paths["class_names.json"]) as f:
    CLASS_NAMES = json.load(f)
DB_LABELS = np.load(paths["db_labels.npy"])
INDEX = faiss.read_index(paths["ivfpq.faissindex"])
DEFAULT_NPROBE = ASSETS["index"]["default_nprobe"]
TRAIN_OFFSETS = np.asarray(ASSETS["train_offsets"])
torch.set_num_threads(NUM_THREADS)
faiss.omp_set_num_threads(NUM_THREADS)

# --- Model: the same CLIP model and preprocessing that produced the index --------------------------
# OpenAI CLIP weights were trained with QuickGELU; force it whenever the weights' config asks for it.
_pretrained = ASSETS["pretrained"] or None
_cfg = open_clip.get_pretrained_cfg(ASSETS["model"], _pretrained) if _pretrained else {}
MODEL, _, PREPROCESS = open_clip.create_model_and_transforms(
    ASSETS["model"], pretrained=_pretrained, force_quick_gelu=bool(_cfg.get("quick_gelu", False)))
# On ZeroGPU the model must be moved to cuda here, at start-up; a real GPU is attached only while a
# @gpu function runs. On CPU hardware this stays on the CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = MODEL.to(DEVICE).eval()


# --- Image lookup: read only the parquet row group that holds a given row ------------------------
class ParquetRows:
    """Reads single rows from the dataset's parquet files, one row group (about 100 images) at a time."""

    def __init__(self):
        self.fs = None if LOCAL_DATA else HfFileSystem()
        self.meta = {}                      # file -> (FileMetaData, row-group start offsets)
        self.cache = OrderedDict()          # (file, row group) -> {"image": [...], "label": [...]}
        self.lock = threading.Lock()

    def _open(self, name):
        if LOCAL_DATA:
            return open(os.path.join(LOCAL_DATA, name), "rb")
        return self.fs.open(f"datasets/{ASSETS['dataset']}/{name}", "rb", block_size=2 * 2**20)

    def metadata(self, name):
        if name not in self.meta:
            with self._open(name) as f:
                md = pq.ParquetFile(f).metadata
            starts = np.concatenate([[0], np.cumsum([md.row_group(g).num_rows for g in range(md.num_row_groups)])])
            self.meta[name] = (md, starts)
        return self.meta[name]

    def row(self, name, row):
        md, starts = self.metadata(name)
        g = int(np.searchsorted(starts, row, side="right") - 1)
        key = (name, g)
        with self.lock:
            group = self.cache.get(key)
            if group is not None:
                self.cache.move_to_end(key)
        if group is None:
            with self._open(name) as f:     # a fresh handle per read: file objects are not thread-safe
                group = pq.ParquetFile(f, metadata=md).read_row_group(g, columns=["image", "label"]).to_pydict()
            with self.lock:
                self.cache[key] = group
                while len(self.cache) > ROWGROUP_CACHE:
                    self.cache.popitem(last=False)
        img = group["image"][row - starts[g]]
        img = img["bytes"] if isinstance(img, dict) else img
        return Image.open(io.BytesIO(img)).convert("RGB"), group["label"][row - starts[g]]

    def train_image(self, gid):
        k = int(np.searchsorted(TRAIN_OFFSETS, gid, side="right") - 1)
        return self.row(ASSETS["train_files"][k], int(gid - TRAIN_OFFSETS[k]))


ROWS = ParquetRows()
POOL = ThreadPoolExecutor(max_workers=max(FETCH_WORKERS, 1))
# Read every file's footer in the background so the first search does not pay for it.
for _name in ASSETS["train_files"] + ASSETS["val_files"]:
    POOL.submit(ROWS.metadata, _name)


def short(c):
    return CLASS_NAMES[int(c)].split(",")[0]


# --- Search ---------------------------------------------------------------------------------------
@gpu(duration=15)
@torch.no_grad()
def embed(img):
    """Only this step uses the GPU; search and image fetching run on the CPU and cost no GPU quota."""
    x = PREPROCESS(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    # fp16 on GPU, as when the database was embedded; fp32 on CPU.
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
        f = MODEL.encode_image(x)
    f = torch.nn.functional.normalize(f.float(), dim=-1)
    return np.ascontiguousarray(f.cpu().numpy(), dtype=np.float32)


def search(img, nprobe):
    if img is None:
        raise gr.Error("Upload an image or pick a random validation image first.")
    t0 = time.perf_counter()
    q = embed(img)
    t1 = time.perf_counter()
    INDEX.nprobe = int(nprobe)
    scores, ids = INDEX.search(q, TOPK)
    t2 = time.perf_counter()
    hits = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
    images = list(POOL.map(lambda h: ROWS.train_image(h[0])[0], hits))
    t3 = time.perf_counter()

    gallery = [(im, f"{rank}. {short(DB_LABELS[gid])} · {score:.3f}")
               for rank, (im, (gid, score)) in enumerate(zip(images, hits), 1)]
    votes = Counter(short(DB_LABELS[gid]) for gid, _ in hits)
    top_class, count = votes.most_common(1)[0]
    summary = (f"**Most common class in the results:** {top_class} ({count}/{len(hits)})\n\n"
               f"Embed {1000 * (t1 - t0):.0f} ms · search {1000 * (t2 - t1):.1f} ms (nprobe {int(nprobe)}) · "
               f"fetch images {1000 * (t3 - t2):.0f} ms")
    return gallery, summary


def random_query():
    """A random ImageNet validation image, streamed from the dataset (never part of the database)."""
    rng = np.random.default_rng()
    name = ASSETS["val_files"][int(rng.integers(len(ASSETS["val_files"])))]
    md, _ = ROWS.metadata(name)
    img, label = ROWS.row(name, int(rng.integers(md.num_rows)))
    return img, f"Random validation image · true class: **{short(label)}**"


# --- Interface ------------------------------------------------------------------------------------
m = ASSETS["measured"]
INTRO = f"""
# ImageNet-1k image search

Upload an image to find the {TOPK} most similar of the **{ASSETS['total_rows']:,} ImageNet-1k training images**.
Images are embedded with CLIP ViT-L/14 and searched with a FAISS **{ASSETS['index']['type']}** index
({m['bytes'] / 1e6:.0f} MB). On 1,000 validation queries at nprobe {DEFAULT_NPROBE}, it reached precision@10
{m['p_at_10']:.3f} and recall@10 {m['recall_at_10']:.3f} against exact search, at {m['lat_median_ms']:.1f} ms
median search time on a 2-core CPU.

**nprobe** is the number of the 4,096 clusters searched: more is more accurate but slower. Each result shows
its rank, its ImageNet class and its cosine similarity to the query.
"""

with gr.Blocks(title="ImageNet image search") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=1):
            query = gr.Image(type="pil", label="Query image", height=300)
            query_note = gr.Markdown()
            nprobe = gr.Dropdown(NPROBE_CHOICES, value=DEFAULT_NPROBE, label="nprobe (clusters searched)")
            with gr.Row():
                random_btn = gr.Button("Random validation image")
                search_btn = gr.Button("Search", variant="primary")
        with gr.Column(scale=2):
            results = gr.Gallery(label=f"Top {TOPK} results", columns=5, height="auto", object_fit="cover")
            summary = gr.Markdown()
    gr.Markdown(
        "Images come from [`" + ASSETS["dataset"] + "`](https://huggingface.co/datasets/" + ASSETS["dataset"] + ") "
        "and are streamed on request, not stored in this Space. ImageNet images are for non-commercial research "
        "and education only.\n\n"
        f"Source code, notebooks and report: [{SOURCE_URL.removeprefix('https://')}]({SOURCE_URL})")

    random_btn.click(random_query, outputs=[query, query_note]).then(
        search, inputs=[query, nprobe], outputs=[results, summary])
    search_btn.click(search, inputs=[query, nprobe], outputs=[results, summary])
    query.upload(lambda: "", outputs=query_note)

if __name__ == "__main__":
    demo.queue(max_size=16).launch()
