    #!/usr/bin/env python3
"""Modern COBOL traceability experiment runner.
Key points
----------
- Correct terminology:
  - TF-IDF = lexical retrieval
  - LSA = latent semantic retrieval
  - BM25 = sparse probabilistic retrieval
  - Embeddings = dense semantic retrieval
- Supports query variants with multiple aggregation strategies.
- Keeps both fixed and adaptive hybrid retrieval.
- Keeps graph reasoning, data-flow features, beam search, and ordering.
- Removes status prediction entirely.
- Supports ablations for retrieval-only, no-graph, no-transition, and no-beam.
Optional dependencies:
- rank_bm25
- sentence-transformers
"""
from __future__ import annotations
import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None
try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import Normalizer
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for this script.") from exc
try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # pragma: no cover
    BM25Okapi = None
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None
# ============================================================
# Configuration
# ============================================================
RANDOM_SEED = 7
DEFAULT_OUTPUT_DIR = "cobol_traceability_outputs_modern"
DEFAULT_TOP_K_FILES = 5
DEFAULT_TOP_K_BLOCKS = 8
DEFAULT_BEAM_WIDTH = 8
DEFAULT_BEAM_SEEDS = 5
DEFAULT_BEAM_JUMPS = 5
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_RETRIEVERS = ["tfidf", "lsa", "bm25", "embed", "hybrid_fixed", "hybrid_adaptive"]
DEFAULT_QUERY_AGGS = ["weighted"]
DEFAULT_ABLATIONS = ["full", "retrieval_only", "no_graph", "no_transition", "no_beam"]
DEFAULT_TRANSITION_MODE = "static"
BOOTSTRAP_SAMPLES = 2000
STRICT_RATIO = 0.62
CONSERVATIVE_RATIO = 0.45
MMR_LAMBDA = 0.78
# fixed hybrid weights (intentionally simple, interpretable)
FIXED_HYBRID_WEIGHTS = {
    "tfidf": 0.27,
    "lsa": 0.27,
    "bm25": 0.24,
    "embed": 0.22,
}
# block ranking weights
BLOCK_FEATURE_WEIGHTS = {
    "retrieval": 0.42,
    "file_prior": 0.12,
    "filename": 0.08,
    "rank_prior": 0.06,
    "structure": 0.12,
    "graph_support": 0.10,
    "dataflow": 0.05,
    "alignment": 0.05,
}
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# ============================================================
# Data classes
# ============================================================
@dataclass
class BlockRecord:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Data structure or service for Block record.
    How to use: Instantiate it where the corresponding workflow needs state or behavior.
    Design note: The class groups related responsibilities so the pipeline remains organized.
    """
    block_id: str
    file_key: str
    label: str
    text: str
    start_line: int
    end_line: int
    index_in_file: int
    is_entry: bool = False
@dataclass(frozen=True)
class RunSpec:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Data structure or service for Run spec.
    How to use: Instantiate it where the corresponding workflow needs state or behavior.
    Design note: The class groups related responsibilities so the pipeline remains organized.
    """
    retriever_mode: str
    query_agg: str
    ablation_mode: str
    transition_mode: str
@dataclass
class RequirementReport:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Data structure or service for Requirement report.
    How to use: Instantiate it where the corresponding workflow needs state or behavior.
    Design note: The class groups related responsibilities so the pipeline remains organized.
    """
    requirement_id: str
    requirement: str
    paraphrases: List[str]
    ground_truth_chain: List[dict]
    ground_truth_evidence: List[dict]
    retrieval: Dict[str, Any]
    block_ranking: Dict[str, Any]
    chain_result: Dict[str, Any]
    metrics: Dict[str, Any]
    debug: Dict[str, Any]
def report_to_dict(report: RequirementReport, spec: Optional[RunSpec] = None, index: Optional[int] = None) -> Dict[str, Any]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Serialize a requirement report into a JSON-friendly dictionary.
    How to use: Use this right before writing logs or summaries to disk.
    Design note: Keeps provenance fields optional so the same report can be reused across runs.
    """
    data: Dict[str, Any] = {
        "requirement_id": report.requirement_id,
        "requirement": report.requirement,
        "paraphrases": report.paraphrases,
        "ground_truth_chain": report.ground_truth_chain,
        "ground_truth_evidence": report.ground_truth_evidence,
        "retrieval": report.retrieval,
        "block_ranking": report.block_ranking,
        "chain_result": report.chain_result,
        "metrics": report.metrics,
        "debug": report.debug,
    }
    if spec is not None:
        data["spec"] = dataclasses.asdict(spec)
    if index is not None:
        data["index"] = index
    return data
def safe_filename(text: str, fallback: str = "item") -> str:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Convert arbitrary text into a filesystem-safe name.
    How to use: Use this when turning requirement IDs or labels into file names.
    Design note: Prevents path issues without changing the semantic identifier.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text or "").strip("._-")
    return cleaned or fallback
def write_requirement_logs(reports: List[RequirementReport], spec: RunSpec, spec_dir: str) -> List[Dict[str, Any]]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Write one structured JSON file per requirement and a JSONL summary.
    How to use: Pass the run reports, spec, and target directory.
    Design note: Keeps experiment artifacts easy to inspect, diff, and archive.
    """
    ensure_dir(spec_dir)
    serializable_reports: List[Dict[str, Any]] = []
    jsonl_path = os.path.join(spec_dir, "requirements.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for idx, report in enumerate(reports, start=1):
            payload = report_to_dict(report, spec=spec, index=idx)
            serializable_reports.append(payload)
            req_id = safe_filename(report.requirement_id or f"REQ_{idx:04d}", fallback=f"REQ_{idx:04d}")
            file_name = f"{idx:04d}_{req_id}.json"
            write_json(os.path.join(spec_dir, file_name), payload)
            jsonl_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    write_json(os.path.join(spec_dir, "requirements_summary.json"), serializable_reports)
    return serializable_reports
# ============================================================
# Utilities
# ============================================================
def ensure_dir(path: str) -> None:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Create an output directory if it does not already exist.
    How to use: Call this before writing any files into a folder.
    Design note: The helper is intentionally silent when the directory already exists.
    """
    os.makedirs(path, exist_ok=True)
def read_json(path: str) -> Any:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Load a JSON document from disk.
    How to use: Use this for datasets, file maps, and saved experiment outputs.
    Design note: Centralizes file I/O so the rest of the script stays clean.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def write_json(path: str, obj: Any) -> None:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Persist a Python object as formatted JSON.
    How to use: Use this for artifacts that need to be human-readable.
    Design note: Indentation is enabled to make inspection and debugging easier.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
def stable_hash(text: str) -> str:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Compute a short stable hash for a text key.
    How to use: Use this when you need a compact deterministic identifier.
    Design note: Helpful for reproducible file names, cache keys, and trace IDs.
    """
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()[:16]
def append_unique(lst: List[str], item: str) -> None:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Append an item only when it is not already present.
    How to use: Use this for ordered de-duplication while preserving sequence.
    Design note: Keeps traces and candidate lists stable across reruns.
    """
    if item not in lst:
        lst.append(item)
def normalize_text(text: str) -> str:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Normalize text for lightweight matching.
    How to use: Use this before lexical comparisons and tokenization.
    Design note: Lowercasing and whitespace cleanup reduce noise without aggressive rewriting.
    """
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()
def tokenize(text: str) -> List[str]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Split normalized text into simple word tokens.
    How to use: Use this for lexical scoring, overlap tests, and BM25 fallbacks.
    Design note: The tokenizer deliberately stays simple and predictable.
    """
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return [t for t in text.split() if t]
def cosine_vec(a: np.ndarray, b: np.ndarray) -> float:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Compute cosine similarity between two numeric vectors.
    How to use: Use this when comparing dense or reduced semantic embeddings.
    Design note: Returns zero safely whenever either vector is empty or near-zero.
    """
    if a is None or b is None:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
def minmax_norm(values: Sequence[float]) -> List[float]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Scale a sequence into the [0, 1] interval.
    How to use: Use this before combining scores from different retrieval models.
    Design note: This makes hybrid aggregation less sensitive to raw score magnitude.
    """
    values = list(values)
    if not values:
        return []
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]
def safe_clip01(x: float) -> float:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Clamp a value into the unit interval.
    How to use: Use this when feature weights must stay bounded.
    Design note: The helper protects downstream scoring from accidental overflow.
    """
    return max(0.0, min(1.0, float(x)))
def overlap_ratio(a: Sequence[str], b: Sequence[str]) -> float:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Measure token overlap from the perspective of the first sequence.
    How to use: Use this for quick lexical recall-style checks.
    Design note: The denominator follows the size of the first input to preserve directional meaning.
    """
    sa, sb = set(a), set(b)
    if not sa:
        return 0.0
    return len(sa & sb) / max(1, len(sa))
def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Compute the longest common subsequence length.
    How to use: Use this when order-sensitive similarity matters.
    Design note: This helper is useful for chain-style comparisons and coarse alignment.
    """
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]
def bootstrap_mean_diff(a: Sequence[float], b: Sequence[float], n_samples: int = BOOTSTRAP_SAMPLES, seed: int = RANDOM_SEED) -> Dict[str, float]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Estimate the mean difference and a simple bootstrap interval.
    How to use: Use this to compare two paired metric series.
    Design note: The result is intentionally lightweight and reproducible.
    """
    if not a or not b or len(a) != len(b):
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}
    rng = np.random.default_rng(seed)
    diffs = np.array(a, dtype=float) - np.array(b, dtype=float)
    base = float(np.mean(diffs))
    idx = rng.integers(0, len(diffs), size=(n_samples, len(diffs)))
    boot = diffs[idx].mean(axis=1)
    ci_low = float(np.quantile(boot, 0.025))
    ci_high = float(np.quantile(boot, 0.975))
    p_value = float(min(1.0, 2.0 * min(np.mean(boot <= 0.0), np.mean(boot >= 0.0))))
    return {"mean_diff": base, "ci_low": ci_low, "ci_high": ci_high, "p_value": p_value}
# ============================================================
# Dataset loading
# ============================================================
def load_dataset_json(dataset_path: str) -> List[dict]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Load a requirement dataset from JSON.
    How to use: Use this for files that expose either a top-level list or a 'requirements' field.
    Design note: The helper validates the format early so failures are explicit.
    """
    data = read_json(dataset_path)
    if isinstance(data, dict) and "requirements" in data:
        return data["requirements"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported dataset format; expected a requirements array.")
def load_files_json(files_path: str) -> Dict[str, str]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Load a file-to-source-code mapping from JSON.
    How to use: Use this for datasets that store code snippets in several common shapes.
    Design note: Supports both direct dictionaries and list-based records.
    """
    data = read_json(files_path)
    if isinstance(data, dict):
        if all(isinstance(v, str) for v in data.values()):
            return data
        if "files" in data and isinstance(data["files"], list):
            out: Dict[str, str] = {}
            for item in data["files"]:
                if isinstance(item, dict):
                    key = item.get("file") or item.get("filename") or item.get("path")
                    code = item.get("code") or item.get("text") or item.get("content")
                    if key and code:
                        out[str(key)] = str(code)
            return out
    if isinstance(data, list):
        out = {}
        for item in data:
            if isinstance(item, dict):
                key = item.get("file") or item.get("filename") or item.get("path")
                code = item.get("code") or item.get("text") or item.get("content")
                if key and code:
                    out[str(key)] = str(code)
        return out
    raise ValueError("Unsupported files.json format.")
# ============================================================
# Normalization / matching
# ============================================================
_EXT_RE = re.compile(r"\.(CBL|COB|CPY|TXT|PY|JAVA|CPP|C|JS|TS|CS|GO|RB|PHP|SQL|XML|JSON|YAML|YML)$", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")
def strip_extension(name: str) -> str:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Remove a known source-code extension from a file name.
    How to use: Use this before normalizing identifiers for lookup.
    Design note: Keeps program names comparable across COBOL and related artifacts.
    """
    base = os.path.basename(name or "").strip()
    return _EXT_RE.sub("", base)
def normalize_hyphen(name: str) -> str:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Normalize an identifier into an uppercase hyphenated key.
    How to use: Use this for approximate matching and graph indexing.
    Design note: This is useful when the same program appears with slightly different punctuation.
    """
    base = strip_extension(name).upper()
    base = _NON_ALNUM_RE.sub("-", base)
    base = re.sub(r"-+", "-", base).strip("-")
    return base
def normalize_compact(name: str) -> str:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Normalize an identifier into an uppercase compact key.
    How to use: Use this for exact-ish canonical keys that should ignore punctuation.
    Design note: Compact keys make map lookups resilient to separators and file naming differences.
    """
    base = strip_extension(name).upper()
    base = _NON_ALNUM_RE.sub("", base)
    return base
def extract_program_id(code: str, fallback_name: str) -> str:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Extract the COBOL program or module identifier when possible.
    How to use: Use this as the canonical logical name for a file.
    Design note: Falls back to the file stem when no explicit identifier is present.
    """
    patterns = [
        r"PROGRAM-ID\.\s*([A-Z0-9\-]+)",
        r"MODULE-ID\.\s*([A-Z0-9\-]+)",
        r"CLASS\s+([A-Z0-9\-]+)",
        r"FUNCTION\s+([A-Z0-9\-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, code or "", re.IGNORECASE)
        if m:
            return normalize_hyphen(m.group(1))
    return normalize_hyphen(os.path.splitext(os.path.basename(fallback_name))[0])
def make_block_id(file_key: str, label: str) -> str:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Build a stable block identifier from file and label information.
    How to use: Use this when referencing a block in metrics or graph structures.
    Design note: The identifier is normalized so it survives formatting variation.
    """
    return f"{normalize_compact(file_key)}::{normalize_compact(label)}"
def normalize_ground_truth_chain(chain: List[dict]) -> List[dict]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Canonicalize ground-truth chain entries.
    How to use: Use this before evaluation so file and block labels have a consistent shape.
    Design note: This keeps scoring logic simpler and less error-prone.
    """
    return [{"file": item.get("file", ""), "block": item.get("block", ""), "role": item.get("role", "")} for item in chain or []]
def gt_files_from_chain(chain: List[dict]) -> Set[str]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Collect the file keys referenced by a ground-truth chain.
    How to use: Use this when computing file-level precision and recall.
    Design note: The output is normalized for robust set comparison.
    """
    return {normalize_compact(item.get("file", "")) for item in chain or [] if item.get("file")}
def gt_block_ids_from_chain(chain: List[dict]) -> List[str]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Convert a ground-truth chain into normalized block identifiers.
    How to use: Use this for block-level recall and order metrics.
    Design note: Only entries with both file and block labels are kept.
    """
    out = []
    for item in chain or []:
        file_key = item.get("file", "")
        block = item.get("block", "")
        if file_key and block:
            out.append(make_block_id(file_key, block))
    return out
def query_variants_from_requirement(req: dict) -> List[str]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Extract the primary requirement text plus optional paraphrases.
    How to use: Use this to build multi-variant queries for retrieval.
    Design note: Deduplication is based on normalized text so paraphrases stay distinct only when they add value.
    """
    variants: List[str] = []
    for key in ["requirement", "original", "query", "text"]:
        val = req.get(key)
        if isinstance(val, str) and val.strip():
            variants.append(val.strip())
            break
    for key in ["paraphrases", "variants", "alternative_phrases", "query_variants"]:
        extra = req.get(key)
        if isinstance(extra, list):
            for item in extra:
                if isinstance(item, str) and item.strip():
                    variants.append(item.strip())
        elif isinstance(extra, str) and extra.strip():
            variants.append(extra.strip())
    seen = set()
    out = []
    for v in variants:
        nv = normalize_text(v)
        if nv and nv not in seen:
            out.append(v)
            seen.add(nv)
    return out
def build_query_specs(req: dict, query_agg: str) -> List[Tuple[str, float]]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Assign weights to the available query variants.
    How to use: Use this before running retrieval so each variant contributes consistently.
    Design note: The helper keeps the original query dominant while still exploiting paraphrases.
    """
    variants = query_variants_from_requirement(req)
    if not variants:
        variants = [req.get("requirement", req.get("original", "")) or ""]
    specs: List[Tuple[str, float]] = []
    for i, q in enumerate(variants):
        base = 1.0 if i == 0 else 0.82
        length_bonus = 0.90 + min(0.20, 0.01 * len(tokenize(q)))
        specs.append((q, base * length_bonus))
    if query_agg == "max":
        return specs
    if query_agg == "weighted":
        return specs
    if query_agg == "softmax":
        return specs
    raise ValueError(f"Unknown query aggregation mode: {query_agg}")
# ============================================================
# COBOL extraction
# ============================================================
HEADER_RE = re.compile(r"^\s*([A-Z0-9][A-Z0-9\-]{2,})\.\s*(?:\*.*)?$", re.IGNORECASE)
SECTION_RE = re.compile(r"^\s*([A-Z0-9][A-Z0-9\-]{2,})\s+SECTION\.\s*(?:\*.*)?$", re.IGNORECASE)
MOVE_CONST_RE = re.compile(r"MOVE\s+(?:'([^']+)'|\"([^\"]+)\")\s+TO\s+([A-Z0-9\-]+)", re.IGNORECASE)
USING_RE = re.compile(r"\bUSING\s+([A-Z0-9\-\s,]+)", re.IGNORECASE)
def extract_blocks_from_code(file_key: str, code: str) -> List[BlockRecord]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Split a source file into coarse executable or structural blocks.
    How to use: Use this to build block-level traceability candidates.
    Design note: When no meaningful block boundaries exist, the full file is used as a fallback.
    """
    lines = (code or "").splitlines()
    if not lines:
        return [BlockRecord(make_block_id(file_key, "FULL"), file_key, "FULL", "", 1, 1, 0, True)]
    blocks: List[BlockRecord] = []
    current_label: Optional[str] = None
    current_start = 1
    current_lines: List[str] = []
    index = 0
    def flush(end_line: int) -> None:
        """
        Author: Venkat Kaushal Thippisetty
        Purpose: Flush.
        How to use: Call this helper as part of the traceability pipeline or supporting utilities.
        Design note: The implementation is documented here to make the codebase easier to read and maintain.
        """
        nonlocal index, current_label, current_lines, current_start
        if current_label is None:
            return
        text = "\n".join(current_lines).strip()
        blocks.append(
            BlockRecord(
                block_id=make_block_id(file_key, current_label),
                file_key=file_key,
                label=current_label,
                text=text,
                start_line=current_start,
                end_line=end_line,
                index_in_file=index,
                is_entry=(index == 0),
            )
        )
        index += 1
    for i, line in enumerate(lines, start=1):
        m = HEADER_RE.match(line) or SECTION_RE.match(line)
        if m:
            if current_label is not None:
                flush(i - 1)
            current_label = normalize_hyphen(m.group(1))
            current_start = i
            current_lines = [line]
        else:
            if current_label is None:
                current_label = "PSEUDO-ENTRY"
                current_start = i
                current_lines = [line]
            else:
                current_lines.append(line)
    if current_label is not None:
        flush(len(lines))
    if len(blocks) <= 1:
        return [BlockRecord(make_block_id(file_key, "FULL"), file_key, "FULL", code or "", 1, max(1, len(lines)), 0, True)]
    cleaned = [b for b in blocks if normalize_compact(b.label) != "PSEUDOENTRY"]
    return cleaned if cleaned else blocks
def extract_copybooks(code: str) -> List[str]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Detect COPY statements and normalize copybook references.
    How to use: Use this when building file-level dependencies from COBOL source.
    Design note: Duplicate references are removed while preserving the first-seen order.
    """
    copybooks = []
    for m in re.finditer(r"\bCOPY\s+([A-Z0-9\-]+)\b", code or "", re.IGNORECASE):
        copybooks.append(normalize_hyphen(m.group(1)))
    for m in re.finditer(r"\bCOPY\s+'([^']+)'", code or "", re.IGNORECASE):
        copybooks.append(normalize_hyphen(m.group(1)))
    seen, out = set(), []
    for c in copybooks:
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out
def extract_call_targets(code: str) -> List[str]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Extract target names from CALL statements.
    How to use: Use this to create inter-file call edges.
    Design note: Quoted and unquoted forms are supported for practical COBOL variants.
    """
    targets = []
    for m in re.finditer(r"\bCALL\s+'([^']+)'", code or "", re.IGNORECASE):
        targets.append(m.group(1))
    for m in re.finditer(r'\bCALL\s+"([^"]+)"', code or "", re.IGNORECASE):
        targets.append(m.group(1))
    for m in re.finditer(r"\bCALL\s+([A-Z0-9\-]+)\b", code or "", re.IGNORECASE):
        raw = m.group(1)
        if raw.upper() not in {"USING", "BY", "VALUE"}:
            targets.append(raw)
    seen, out = set(), []
    for t in targets:
        nk = normalize_hyphen(t)
        if nk and nk not in seen:
            out.append(nk)
            seen.add(nk)
    return out
def extract_link_targets(code: str) -> List[str]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Extract LINK PROGRAM targets.
    How to use: Use this to create explicit inter-program edges.
    Design note: The output is normalized to match graph lookup conventions.
    """
    targets = []
    for m in re.finditer(r"LINK\s+PROGRAM\(\s*'([^']+)'\s*\)", code or "", re.IGNORECASE):
        targets.append(normalize_hyphen(m.group(1)))
    for m in re.finditer(r"LINK\s+PROGRAM\(\s*([A-Z0-9\-]+)\s*\)", code or "", re.IGNORECASE):
        targets.append(normalize_hyphen(m.group(1)))
    seen, out = set(), []
    for t in targets:
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out
def extract_perform_targets(code: str) -> List[str]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Extract candidate labels from PERFORM statements.
    How to use: Use this for intra-file control-flow and paragraph linking.
    Design note: Common COBOL keywords are filtered so only likely targets remain.
    """
    targets = []
    for m in re.finditer(r"\bPERFORM\s+([A-Z0-9\-]+)\b", code or "", re.IGNORECASE):
        t = m.group(1).upper()
        if t in {"THRU", "THROUGH", "VARYING", "UNTIL", "TIMES", "USING", "WITH", "TEST", "AFTER", "BEFORE", "DEPENDING", "ON", "FROM", "TO"}:
            continue
        targets.append(normalize_hyphen(t))
    seen, out = set(), []
    for t in targets:
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out
def extract_move_constants(code: str) -> Dict[str, str]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Map variables that receive literal MOVE values.
    How to use: Use this to approximate data-flow dependencies.
    Design note: The mapping is intentionally conservative and only captures simple literal assignments.
    """
    mapping: Dict[str, str] = {}
    if not code:
        return mapping
    for m in MOVE_CONST_RE.finditer(code):
        literal = m.group(1) or m.group(2) or ""
        var = m.group(3) or ""
        if literal and var:
            mapping[normalize_hyphen(var)] = normalize_hyphen(literal)
    return mapping
# ============================================================
# File views
# ============================================================
def build_file_view(file_key: str, code: str) -> str:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Create a human-readable summary view of a file.
    How to use: Use this for debugging, logging, or lightweight model prompts.
    Design note: Large files are truncated into head and tail snippets to keep context manageable.
    """
    lines = (code or "").splitlines()
    if not lines:
        return f"FILE: {file_key}\nEMPTY FILE."
    def numbered(sub_lines: List[str], start: int = 1) -> str:
        """
        Author: Venkat Kaushal Thippisetty
        Purpose: Numbered.
        How to use: Call this helper as part of the traceability pipeline or supporting utilities.
        Design note: The implementation is documented here to make the codebase easier to read and maintain.
        """
        return "\n".join(f"{i:04d}: {line}" for i, line in enumerate(sub_lines, start=start))
    if len(code) <= 14000 and len(lines) <= 220:
        return f"FILE: {file_key}\nFULL FILE:\n{numbered(lines, 1)}"
    head = lines[:160]
    tail = lines[-80:] if len(lines) > 80 else []
    hints = [
        f"FILE: {file_key}",
        f"APPROX_LINES: {len(lines)}",
        f"PROGRAM_ID: {extract_program_id(code, file_key)}",
        f"COPYBOOKS: {', '.join(extract_copybooks(code)) or 'None'}",
        f"CALL_TARGETS: {', '.join(extract_call_targets(code)[:20]) or 'None'}",
        f"LINK_TARGETS: {', '.join(extract_link_targets(code)[:20]) or 'None'}",
        f"PERFORM_TARGETS: {', '.join(extract_perform_targets(code)[:20]) or 'None'}",
        "HEAD SNIPPET:",
        numbered(head, 1),
    ]
    if tail:
        hints += ["TAIL SNIPPET:", numbered(tail, max(1, len(lines) - len(tail) + 1))]
    return "\n".join(hints)
# ============================================================
# Retrieval models
# ============================================================
class TFIDFSpace:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Represent a lexical TF-IDF retrieval space.
    How to use: Instantiate this with a corpus and then call score(query).
    Design note: It provides a sparse lexical baseline for traceability retrieval.
    """
    def __init__(self, corpus: List[str], item_keys: List[str]):
        """
        Author: Venkat Kaushal Thippisetty
        Purpose:   init  .
        How to use: Call this helper as part of the traceability pipeline or supporting utilities.
        Design note: The implementation is documented here to make the codebase easier to read and maintain.
        """
        self.item_keys = item_keys
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=20000)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus if corpus else [""])
    def score(self, query: str) -> np.ndarray:
        """
        Author: Venkat Kaushal Thippisetty
        Purpose: Score.
        How to use: Call this helper as part of the traceability pipeline or supporting utilities.
        Design note: The implementation is documented here to make the codebase easier to read and maintain.
        """
        q = self.vectorizer.transform([query])
        return (self.tfidf_matrix @ q.T).toarray().reshape(-1)
class LSASpace:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Represent a latent semantic retrieval space built from TF-IDF.
    How to use: Instantiate this on top of a TF-IDF space and then call score(query).
    Design note: The class offers a compact semantic baseline without requiring external embeddings.
    """
    def __init__(self, tfidf_space: TFIDFSpace):
        """
        Author: Venkat Kaushal Thippisetty
        Purpose:   init  .
        How to use: Call this helper as part of the traceability pipeline or supporting utilities.
        Design note: The implementation is documented here to make the codebase easier to read and maintain.
        """
        self.item_keys = tfidf_space.item_keys
        self.vectorizer = tfidf_space.vectorizer
        self.tfidf_matrix = tfidf_space.tfidf_matrix
        n_features = self.tfidf_matrix.shape[1]
        n_samples = self.tfidf_matrix.shape[0]
        n_components = max(2, min(128, n_features - 1 if n_features > 1 else 1, n_samples - 1 if n_samples > 1 else 1))
        self.svd: Optional[TruncatedSVD] = None
        self.normalizer: Optional[Normalizer] = None
        if n_components >= 2 and self.tfidf_matrix.shape[0] > 2 and self.tfidf_matrix.shape[1] > 2:
            self.svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
            lsa = self.svd.fit_transform(self.tfidf_matrix)
            self.normalizer = Normalizer(copy=False)
            self.lsa_matrix = self.normalizer.fit_transform(lsa)
        else:
            self.lsa_matrix = self.tfidf_matrix.toarray().astype(float)
    def score(self, query: str) -> np.ndarray:
        """
        Author: Venkat Kaushal Thippisetty
        Purpose: Score.
        How to use: Call this helper as part of the traceability pipeline or supporting utilities.
        Design note: The implementation is documented here to make the codebase easier to read and maintain.
        """
        q_tfidf = self.vectorizer.transform([query])
        if self.svd is not None:
            q_lsa = self.svd.transform(q_tfidf)
            if self.normalizer is not None:
                q_lsa = self.normalizer.transform(q_lsa)
            q_vec = q_lsa[0]
            return np.array([cosine_vec(q_vec, row) for row in self.lsa_matrix], dtype=float)
        return (self.tfidf_matrix @ q_tfidf.T).toarray().reshape(-1)
class BM25Space:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Represent a BM25-style sparse retrieval space.
    How to use: Instantiate this with tokenized corpus text and then call score(query).
    Design note: When the optional dependency is missing, a simple overlap-based fallback is used.
    """
    def __init__(self, corpus: List[str], item_keys: List[str]):
        """
        Author: Venkat Kaushal Thippisetty
        Purpose:   init  .
        How to use: Call this helper as part of the traceability pipeline or supporting utilities.
        Design note: The implementation is documented here to make the codebase easier to read and maintain.
        """
        self.item_keys = item_keys
        self.tokens = [tokenize(text) for text in corpus]
        self.model = BM25Okapi(self.tokens) if BM25Okapi is not None else None
    def score(self, query: str) -> np.ndarray:
        """
        Author: Venkat Kaushal Thippisetty
        Purpose: Score.
        How to use: Call this helper as part of the traceability pipeline or supporting utilities.
        Design note: The implementation is documented here to make the codebase easier to read and maintain.
        """
        q_tokens = tokenize(query)
        if self.model is None:
            qset = set(q_tokens)
            vals = []
            for doc_tokens in self.tokens:
                dset = set(doc_tokens)
                vals.append(len(qset & dset) / max(1, len(qset)))
            return np.array(vals, dtype=float)
        return np.array(self.model.get_scores(q_tokens), dtype=float)
class EmbeddingSpace:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Represent a dense semantic embedding retrieval space.
    How to use: Instantiate this with a transformer model name and then call score(query).
    Design note: A TF-IDF fallback keeps the script usable when sentence-transformers is unavailable.
    """
    def __init__(self, corpus: List[str], item_keys: List[str], model_name: str):
        """
        Author: Venkat Kaushal Thippisetty
        Purpose:   init  .
        How to use: Call this helper as part of the traceability pipeline or supporting utilities.
        Design note: The implementation is documented here to make the codebase easier to read and maintain.
        """
        self.item_keys = item_keys
        self.model_name = model_name
        self.available = False
        self.model = None
        self.fallback_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=20000)
        self.fallback_matrix = self.fallback_vectorizer.fit_transform(corpus if corpus else [""])
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name)
                self.available = True
                self.embeddings = np.asarray(self.model.encode(corpus if corpus else [""], normalize_embeddings=True, show_progress_bar=False), dtype=float)
            except Exception:
                self.available = False
                self.model = None
                self.embeddings = None
        else:
            self.embeddings = None
    def score(self, query: str) -> np.ndarray:
        """
        Author: Venkat Kaushal Thippisetty
        Purpose: Score.
        How to use: Call this helper as part of the traceability pipeline or supporting utilities.
        Design note: The implementation is documented here to make the codebase easier to read and maintain.
        """
        if self.available and self.model is not None and self.embeddings is not None:
            q = np.asarray(self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0], dtype=float)
            return np.dot(self.embeddings, q)
        q = self.fallback_vectorizer.transform([query])
        return (self.fallback_matrix @ q.T).toarray().reshape(-1)
class RetrievalSuite:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Bundle all retrieval back ends and their hybrid scoring logic.
    How to use: Instantiate this once per corpus and then query it repeatedly.
    Design note: This centralizes lexical, latent, sparse, and dense retrieval in one place.
    """
    def __init__(self, corpus: List[str], item_keys: List[str], embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        """
        Author: Venkat Kaushal Thippisetty
        Purpose:   init  .
        How to use: Call this helper as part of the traceability pipeline or supporting utilities.
        Design note: The implementation is documented here to make the codebase easier to read and maintain.
        """
        self.item_keys = item_keys
        self.tfidf = TFIDFSpace(corpus, item_keys)
        self.lsa = LSASpace(self.tfidf)
        self.bm25 = BM25Space(corpus, item_keys)
        self.embed = EmbeddingSpace(corpus, item_keys, embedding_model)
    def _component_scores(self, query: str) -> Dict[str, np.ndarray]:
        """
        Author: Venkat Kaushal Thippisetty
        Purpose:  component scores.
        How to use: Call this helper as part of the traceability pipeline or supporting utilities.
        Design note: The implementation is documented here to make the codebase easier to read and maintain.
        """
        return {
            "tfidf": np.asarray(self.tfidf.score(query), dtype=float),
            "lsa": np.asarray(self.lsa.score(query), dtype=float),
            "bm25": np.asarray(self.bm25.score(query), dtype=float),
            "embed": np.asarray(self.embed.score(query), dtype=float),
        }
    def _adaptive_hybrid_weights(self, components: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Author: Venkat Kaushal Thippisetty
        Purpose:  adaptive hybrid weights.
        How to use: Call this helper as part of the traceability pipeline or supporting utilities.
        Design note: The implementation is documented here to make the codebase easier to read and maintain.
        """
        base = dict(FIXED_HYBRID_WEIGHTS)
        confidences: Dict[str, float] = {}
        for name, arr in components.items():
            a = np.asarray(arr, dtype=float)
            if a.size == 0:
                confidences[name] = 0.0
                continue
            if a.size == 1:
                top1 = float(a[0])
                gap = top1
            else:
                part = np.partition(a, -2)[-2:]
                top1 = float(np.max(part))
                top2 = float(np.min(part))
                gap = max(0.0, top1 - top2)
            spread = float(np.std(a))
            confidences[name] = max(0.0, 0.60 * top1 + 0.30 * gap + 0.10 * spread)
        raw = {k: base[k] * (0.75 + confidences.get(k, 0.0)) for k in base}
        s = sum(raw.values()) or 1.0
        return {k: v / s for k, v in raw.items()}
    def aggregate_queries(self, query_specs: Sequence[Tuple[str, float]], query_agg: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Author: Venkat Kaushal Thippisetty
        Purpose: Aggregate queries.
        How to use: Call this helper as part of the traceability pipeline or supporting utilities.
        Design note: The implementation is documented here to make the codebase easier to read and maintain.
        """
        if not query_specs:
            query_specs = [("", 1.0)]
        variant_components: List[Dict[str, np.ndarray]] = []
        weights = []
        debug_variants: List[Dict[str, Any]] = []
        for q, weight in query_specs:
            comps = self._component_scores(q)
            # normalize each query's component scores so no variant dominates by scale only
            comps = {k: np.asarray(minmax_norm(v.tolist()), dtype=float) for k, v in comps.items()}
            variant_components.append(comps)
            weights.append(float(weight))
            debug_variants.append({"query": q, "weight": weight})
        weights = np.asarray(weights, dtype=float)
        if float(weights.sum()) <= 0.0:
            weights = np.ones_like(weights)
        norm_weights = weights / weights.sum()
        agg: Dict[str, np.ndarray] = {"tfidf": None, "lsa": None, "bm25": None, "embed": None}  # type: ignore[assignment]
        for name in agg:
            mats = np.vstack([vc[name] * w for vc, w in zip(variant_components, weights)])
            if query_agg == "max":
                agg[name] = np.max(mats, axis=0)
            elif query_agg == "weighted":
                agg[name] = np.average(np.vstack([vc[name] for vc in variant_components]), axis=0, weights=norm_weights)
            elif query_agg == "softmax":
                temperature = 0.75
                sw = np.exp(norm_weights / max(1e-6, temperature))
                sw = sw / sw.sum()
                agg[name] = np.average(np.vstack([vc[name] for vc in variant_components]), axis=0, weights=sw)
            else:
                raise ValueError(f"Unknown query aggregation mode: {query_agg}")
        debug = {"variants": debug_variants, "query_agg": query_agg}
        return agg, debug
    def combine(self, components: Dict[str, np.ndarray], retriever_mode: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Author: Venkat Kaushal Thippisetty
        Purpose: Combine.
        How to use: Call this helper as part of the traceability pipeline or supporting utilities.
        Design note: The implementation is documented here to make the codebase easier to read and maintain.
        """
        if retriever_mode == "tfidf":
            return np.asarray(components["tfidf"], dtype=float), {"mode": "tfidf"}
        if retriever_mode == "lsa":
            return np.asarray(components["lsa"], dtype=float), {"mode": "lsa"}
        if retriever_mode == "bm25":
            return np.asarray(components["bm25"], dtype=float), {"mode": "bm25"}
        if retriever_mode == "embed":
            return np.asarray(components["embed"], dtype=float), {"mode": "embed"}
        if retriever_mode == "hybrid_fixed":
            final = (
                FIXED_HYBRID_WEIGHTS["tfidf"] * components["tfidf"]
                + FIXED_HYBRID_WEIGHTS["lsa"] * components["lsa"]
                + FIXED_HYBRID_WEIGHTS["bm25"] * components["bm25"]
                + FIXED_HYBRID_WEIGHTS["embed"] * components["embed"]
            )
            return np.asarray(final, dtype=float), dict(FIXED_HYBRID_WEIGHTS)
        if retriever_mode == "hybrid_adaptive":
            weights = self._adaptive_hybrid_weights(components)
            final = (
                weights["tfidf"] * components["tfidf"]
                + weights["lsa"] * components["lsa"]
                + weights["bm25"] * components["bm25"]
                + weights["embed"] * components["embed"]
            )
            return np.asarray(final, dtype=float), weights
        raise ValueError(f"Unknown retriever mode: {retriever_mode}")
    def score(self, query_specs: Sequence[Tuple[str, float]], query_agg: str, retriever_mode: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Author: Venkat Kaushal Thippisetty
        Purpose: Score.
        How to use: Call this helper as part of the traceability pipeline or supporting utilities.
        Design note: The implementation is documented here to make the codebase easier to read and maintain.
        """
        components, query_debug = self.aggregate_queries(query_specs, query_agg)
        final, retriever_debug = self.combine(components, retriever_mode)
        debug = {
            "components": {k: v.tolist() for k, v in components.items()},
            "query_debug": query_debug,
            "retriever_debug": retriever_debug,
        }
        return np.asarray(final, dtype=float), debug
# ============================================================
# Graph
# ============================================================
class TraceGraph:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Build and query file-level and block-level trace graphs.
    How to use: Instantiate this from the loaded files and extracted blocks.
    Design note: The graph supports both dependency discovery and chain reconstruction.
    """
    def __init__(self, files: Dict[str, str], blocks_by_file: Dict[str, List[BlockRecord]]):
        """
        Author: Venkat Kaushal Thippisetty
        Purpose:   init  .
        How to use: Call this helper as part of the traceability pipeline or supporting utilities.
        Design note: The implementation is documented here to make the codebase easier to read and maintain.
        """
        self.files = files
        self.blocks_by_file = blocks_by_file
        self.file_index: Dict[str, str] = {}
        self.file_graph_forward = defaultdict(set)
        self.file_graph_backward = defaultdict(set)
        self.block_graph_forward = defaultdict(set)
        self.block_graph_backward = defaultdict(set)
        self.file_entry_block: Dict[str, str] = {}
        self.block_index: Dict[str, BlockRecord] = {}
        self.report = {
            "file_nodes": 0,
            "block_nodes": 0,
            "file_call_edges": 0,
            "file_link_edges": 0,
            "block_internal_edges": 0,
            "block_call_edges": 0,
            "block_link_edges": 0,
            "block_dataflow_edges": 0,
        }
        self._build()
    def _build(self) -> None:
        """
        Author: Venkat Kaushal Thippisetty
        Purpose:  build.
        How to use: Call this helper as part of the traceability pipeline or supporting utilities.
        Design note: The implementation is documented here to make the codebase easier to read and maintain.
        """
        for file_key, code in self.files.items():
            stem = normalize_hyphen(os.path.splitext(os.path.basename(file_key))[0])
            prog = extract_program_id(code, file_key)
            for cand in [stem, normalize_hyphen(file_key), normalize_compact(file_key), prog, normalize_compact(prog)]:
                if cand:
                    self.file_index[cand] = file_key
        self.report["file_nodes"] = len(self.files)
        for file_key, blocks in self.blocks_by_file.items():
            if blocks:
                self.file_entry_block[file_key] = blocks[0].block_id
            for b in blocks:
                self.block_index[b.block_id] = b
        self.report["block_nodes"] = len(self.block_index)
        for src_file, code in self.files.items():
            dynamic_map = extract_move_constants(code)
            for raw in extract_call_targets(code):
                dst = self.resolve_file(raw)
                if dst is None and raw in dynamic_map:
                    dst = self.resolve_file(dynamic_map[raw])
                if dst:
                    self.file_graph_forward[src_file].add(dst)
                    self.file_graph_backward[dst].add(src_file)
                    self.report["file_call_edges"] += 1
            for raw in extract_link_targets(code):
                dst = self.resolve_file(raw)
                if dst:
                    self.file_graph_forward[src_file].add(dst)
                    self.file_graph_backward[dst].add(src_file)
                    self.report["file_link_edges"] += 1
        for file_key, blocks in self.blocks_by_file.items():
            label_to_block = {normalize_hyphen(b.label): b.block_id for b in blocks}
            var_blocks: Dict[str, List[str]] = defaultdict(list)
            for i, b in enumerate(blocks):
                if i + 1 < len(blocks):
                    nxt = blocks[i + 1].block_id
                    self.block_graph_forward[b.block_id].add(nxt)
                    self.block_graph_backward[nxt].add(b.block_id)
                    self.report["block_internal_edges"] += 1
                for tgt in extract_perform_targets(b.text):
                    tgt_norm = normalize_hyphen(tgt)
                    if tgt_norm in label_to_block:
                        tid = label_to_block[tgt_norm]
                        self.block_graph_forward[b.block_id].add(tid)
                        self.block_graph_backward[tid].add(b.block_id)
                        self.report["block_internal_edges"] += 1
                for raw in extract_call_targets(b.text):
                    dst_file = self.resolve_file(raw)
                    if dst_file and dst_file in self.file_entry_block:
                        tid = self.file_entry_block[dst_file]
                        self.block_graph_forward[b.block_id].add(tid)
                        self.block_graph_backward[tid].add(b.block_id)
                        self.report["block_call_edges"] += 1
                for raw in extract_link_targets(b.text):
                    dst_file = self.resolve_file(raw)
                    if dst_file and dst_file in self.file_entry_block:
                        tid = self.file_entry_block[dst_file]
                        self.block_graph_forward[b.block_id].add(tid)
                        self.block_graph_backward[tid].add(b.block_id)
                        self.report["block_link_edges"] += 1
                moves = extract_move_constants(b.text)
                for var, const in moves.items():
                    var_blocks[var].append(b.block_id)
                    dst = self.resolve_file(const)
                    if dst and dst in self.file_entry_block:
                        tid = self.file_entry_block[dst]
                        self.block_graph_forward[b.block_id].add(tid)
                        self.block_graph_backward[tid].add(b.block_id)
                        self.report["block_dataflow_edges"] += 1
                for m in USING_RE.finditer(b.text or ""):
                    vars_part = m.group(1) or ""
                    for tok in re.split(r"[\s,]+", vars_part):
                        tok = normalize_hyphen(tok)
                        if tok:
                            var_blocks[tok].append(b.block_id)
            for _var, bids in var_blocks.items():
                uniq: List[str] = []
                for bid in bids:
                    append_unique(uniq, bid)
                if len(uniq) > 1:
                    for i in range(len(uniq) - 1):
                        a, c = uniq[i], uniq[i + 1]
                        self.block_graph_forward[a].add(c)
                        self.block_graph_backward[c].add(a)
                        self.report["block_dataflow_edges"] += 1
    def resolve_file(self, raw: str) -> Optional[str]:
        """
        Author: Venkat Kaushal Thippisetty
        Purpose: Resolve file.
        How to use: Call this helper as part of the traceability pipeline or supporting utilities.
        Design note: The implementation is documented here to make the codebase easier to read and maintain.
        """
        if not raw:
            return None
        for candidate in [normalize_hyphen(raw), normalize_compact(raw), os.path.splitext(os.path.basename(raw))[0]]:
            if candidate in self.file_index:
                return self.file_index[candidate]
        return None
    def file_neighbors(self, file_key: str) -> List[str]:
        """
        Author: Venkat Kaushal Thippisetty
        Purpose: Return adjacent file nodes in the inter-file graph.
        How to use: Use this when expanding retrieval candidates through graph reasoning.
        Design note: Neighbors are collected from both forward and backward edges.
        """
        return sorted(self.file_graph_forward.get(file_key, set()) | self.file_graph_backward.get(file_key, set()))
    def block_neighbors(self, block_id: str) -> List[str]:
        """
        Author: Venkat Kaushal Thippisetty
        Purpose: Return adjacent block nodes in the block graph.
        How to use: Use this when expanding or smoothing block scores.
        Design note: The graph is treated as undirected for neighborhood lookup.
        """
        return sorted(self.block_graph_forward.get(block_id, set()) | self.block_graph_backward.get(block_id, set()))
    def block_out_neighbors(self, block_id: str) -> List[str]:
        """
        Author: Venkat Kaushal Thippisetty
        Purpose: Return outgoing block neighbors only.
        How to use: Use this when chain reconstruction must respect directional flow.
        Design note: This is the stricter view used by beam search.
        """
        return sorted(self.block_graph_forward.get(block_id, set()))
    def block_in_degree(self, block_id: str) -> int:
        """
        Author: Venkat Kaushal Thippisetty
        Purpose: Compute the incoming degree of a block.
        How to use: Use this to identify likely entry points in a file.
        Design note: Zero in-degree blocks are treated as starting candidates.
        """
        return len(self.block_graph_backward.get(block_id, set()))
# ============================================================
# Feature helpers
# ============================================================
def retrieval_rank_feature(rank: int, top_k: int) -> float:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Convert a retrieval rank into a bounded feature value.
    How to use: Use this when blending rank priors into block scoring.
    Design note: Higher-ranked items receive a stronger contribution.
    """
    if top_k <= 1:
        return 1.0
    return max(0.0, (top_k - rank + 1) / top_k)
def filename_feature(query: str, file_key: str) -> float:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Measure how much the requirement text resembles the file name.
    How to use: Use this as a lightweight lexical prior.
    Design note: This helps surface files whose names already hint at the target behavior.
    """
    return overlap_ratio(tokenize(query), tokenize(os.path.splitext(os.path.basename(file_key))[0]))
def file_structure_feature(file_view: str) -> float:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Score coarse file structure cues in a file view.
    How to use: Use this for high-level file ranking and explainability.
    Design note: Presence of division, sections, and statement patterns increases confidence.
    """
    lowered = file_view.lower()
    score = 0.0
    score += 0.25 if "program-id" in lowered else 0.0
    score += 0.20 if "procedure division" in lowered else 0.0
    score += 0.12 if "working-storage section" in lowered else 0.0
    score += 0.10 if "exec sql" in lowered else 0.0
    score += 0.10 if "exec cics" in lowered else 0.0
    score += 0.08 if "call " in lowered else 0.0
    score += 0.08 if "perform " in lowered else 0.0
    return min(score, 1.0)
def block_structure_feature(block: BlockRecord) -> float:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Score structural cues inside a block.
    How to use: Use this to favor blocks that contain executable or database-like operations.
    Design note: This is a weak heuristic, not a proof of relevance.
    """
    lowered = block.text.lower()
    score = 0.0
    score += 0.25 if "exec sql" in lowered else 0.0
    score += 0.25 if "exec cics" in lowered else 0.0
    score += 0.20 if "perform " in lowered else 0.0
    score += 0.20 if "call " in lowered else 0.0
    score += 0.10 if "read " in lowered else 0.0
    score += 0.10 if "write " in lowered else 0.0
    score += 0.10 if "delete " in lowered else 0.0
    score += 0.10 if "insert " in lowered else 0.0
    return min(score, 1.0)
def dataflow_feature(block: BlockRecord, graph: TraceGraph) -> float:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Estimate graph-based data-flow salience for a block.
    How to use: Use this to emphasize blocks with richer dependency context.
    Design note: The score is intentionally small so it supplements rather than dominates retrieval.
    """
    out_deg = len(graph.block_out_neighbors(block.block_id))
    in_deg = graph.block_in_degree(block.block_id)
    return min(1.0, 0.06 * out_deg + 0.04 * in_deg)
def requirement_alignment_score(query: str, block_text: str) -> float:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Estimate lexical alignment between the requirement and a code block.
    How to use: Use this before final ranking or chain reconstruction.
    Design note: This is a simple readability-oriented heuristic rather than a learned semantic model.
    """
    q = tokenize(query)
    b = tokenize(block_text)
    if not q or not b:
        return 0.0
    j = overlap_ratio(q, b)
    return min(1.0, 0.75 * j + 0.25 * (len(set(q) & set(b)) / max(1, len(set(q)))))
def diversity_penalty(selected_texts: List[str], current_text: str) -> float:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Estimate how redundant a candidate text is relative to selections already made.
    How to use: Use this to avoid returning nearly identical blocks.
    Design note: Lower diversity penalties keep selected evidence broader and more useful.
    """
    if not selected_texts:
        return 0.0
    cur = set(tokenize(current_text))
    if not cur:
        return 0.0
    sims = []
    for text in selected_texts:
        prev = set(tokenize(text))
        if not prev:
            continue
        inter = len(cur & prev)
        union = len(cur | prev)
        if union:
            sims.append(inter / union)
    return max(sims) if sims else 0.0
def combine_block_features(features: Dict[str, float]) -> float:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Merge normalized block features into one final score.
    How to use: Use this after feature construction and normalization.
    Design note: The fixed weights keep the ranking transparent and easy to tune.
    """
    score = 0.0
    for name, weight in BLOCK_FEATURE_WEIGHTS.items():
        score += weight * safe_clip01(features.get(name, 0.0))
    return score
# ============================================================
# Chain reconstruction
# ============================================================
def _numeric_prefix(label: str) -> Optional[int]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Extract a leading numeric prefix from a label when present.
    How to use: Use this for rough ordering comparisons between paragraphs or sections.
    Design note: Returns None when the label is not numeric-prefixed.
    """
    if not label:
        return None
    m = re.match(r"^\s*(\d{2,6})", label)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None
def _transition_bonus(prev_block: BlockRecord, next_block: BlockRecord, graph: TraceGraph, transition_mode: str, use_block_graph: bool) -> float:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Compute the local bonus for moving from one block to the next.
    How to use: Use this inside chain reconstruction to reward plausible transitions.
    Design note: The score combines graph adjacency, file locality, and ordering cues.
    """
    if prev_block.block_id == next_block.block_id:
        return -1.0
    bonus = 0.0
    bonus += 0.05 if prev_block.file_key == next_block.file_key else 0.02
    if use_block_graph and next_block.block_id in set(graph.block_out_neighbors(prev_block.block_id)):
        bonus += 0.45
    prev_num = _numeric_prefix(prev_block.label)
    next_num = _numeric_prefix(next_block.label)
    if prev_num is not None and next_num is not None:
        gap = abs(next_num - prev_num)
        bonus += max(0.0, 0.16 - min(gap, 2000) / 2000.0 * 0.16)
    if next_block.is_entry and not prev_block.is_entry:
        bonus -= 0.04
    if transition_mode == "adaptive":
        local_degree = len(graph.block_out_neighbors(prev_block.block_id)) + len(graph.block_out_neighbors(next_block.block_id))
        scale = 1.0 / (1.0 + math.log1p(local_degree))
        bonus *= (0.85 + 0.30 * scale)
    return bonus
# def reconstruct_chain(
#     ranked_blocks: List[Tuple[str, float, BlockRecord]],
#     graph: TraceGraph,
#     max_steps: int,
#     requirement_text: str,
#     use_beam: bool = True,
#     use_block_graph: bool = True,
#     use_diversity: bool = True,
#     use_requirement_alignment: bool = True,
#     use_dataflow: bool = True,
#     transition_mode: str = "static",
#     beam_width: int = DEFAULT_BEAM_WIDTH,
#     seed_count: int = DEFAULT_BEAM_SEEDS,
#     jump_k: int = DEFAULT_BEAM_JUMPS,
#     beam_length_alpha: float = 0.65,
#     use_transition: bool = True,
# ) -> List[str]:
#     if not ranked_blocks:
#         return []
#     normalized = sorted(ranked_blocks, key=lambda x: x[1], reverse=True)
#     score_map = {bid: score for bid, score, _ in normalized}
#     block_map = {bid: block for bid, score, block in normalized}
#     ordered_ids = [bid for bid, _, _ in normalized]
#     entry_nodes = {bid for bid, _, _ in normalized if graph.block_in_degree(bid) == 0}
#     if not use_beam:
#         return ordered_ids[:max_steps]
#     def beam_value(state) -> float:
#         ln = max(1, len(state["chain"]))
#         return state["raw_score"] / math.pow(ln, beam_length_alpha)
#     beams = []
#     for bid, score, block in normalized[: max(1, seed_count)]:
#         bonus = 0.12 if bid in entry_nodes else 0.0
#         beams.append({"chain": [bid], "visited": {bid}, "raw_score": score + bonus, "last": bid, "files": {block.file_key}})
#     for bid in list(entry_nodes)[: max(1, seed_count)]:
#         if bid not in {s["last"] for s in beams}:
#             beams.append({"chain": [bid], "visited": {bid}, "raw_score": score_map.get(bid, 0.0) + 0.12, "last": bid, "files": {block_map[bid].file_key}})
#     if not beams:
#         first = ordered_ids[0]
#         beams = [{"chain": [first], "visited": {first}, "raw_score": score_map.get(first, 0.0), "last": first, "files": {block_map[first].file_key}}]
#     best_state = max(beams, key=beam_value)
#     for _ in range(1, max_steps):
#         expanded = []
#         for state in beams:
#             last_id = state["last"]
#             last_block = block_map[last_id]
#             successors: List[str] = []
#             if use_block_graph:
#                 successors = [n for n in graph.block_out_neighbors(last_id) if n in score_map and n not in state["visited"]]
#             if not successors:
#                 successors = [bid for bid in ordered_ids if bid not in state["visited"]][:jump_k]
#             else:
#                 for bid in [bid for bid in ordered_ids if bid not in state["visited"]][:jump_k]:
#                     if bid not in successors:
#                         successors.append(bid)
#             for bid in successors:
#                 next_block = block_map[bid]
#                 trans = 0.0
#                 if use_transition:
#                     trans = _transition_bonus(last_block, next_block, graph, transition_mode, use_block_graph)
#                 jump_penalty = -0.03 if bid not in set(graph.block_out_neighbors(last_id)) else 0.0
#                 file_bonus = 0.03 if next_block.file_key == last_block.file_key else 0.0
#                 align = requirement_alignment_score(requirement_text, next_block.text) if use_requirement_alignment else 0.0
#                 dataflow = dataflow_feature(next_block, graph) if use_dataflow else 0.0
#                 new_state = {
#                     "chain": state["chain"] + [bid],
#                     "visited": set(state["visited"]) | {bid},
#                     "raw_score": state["raw_score"] + score_map.get(bid, 0.0) + trans + jump_penalty + file_bonus + 0.20 * align + 0.10 * dataflow,
#                     "last": bid,
#                     "files": set(state["files"]) | {next_block.file_key},
#                 }
#                 expanded.append(new_state)
#         if not expanded:
#             break
#         expanded.sort(key=lambda s: beam_value(s), reverse=True)
#         if use_diversity:
#             pruned = []
#             seen_suffixes = set()
#             for s in expanded:
#                 suffix = tuple(s["chain"][-2:]) if len(s["chain"]) >= 2 else tuple(s["chain"])
#                 if suffix not in seen_suffixes:
#                     pruned.append(s)
#                     seen_suffixes.add(suffix)
#                 if len(pruned) >= beam_width:
#                     break
#             beams = pruned if pruned else expanded[:beam_width]
#         else:
#             beams = expanded[:beam_width]
#         current_best = max(beams, key=beam_value)
#         if beam_value(current_best) > beam_value(best_state):
#             best_state = current_best
#     best_state = max(beams + [best_state], key=beam_value)
#     chain = best_state["chain"][:max_steps]
#     if len(chain) < min(max_steps, 2):
#         for bid in ordered_ids:
#             if bid not in chain:
#                 chain.append(bid)
#             if len(chain) >= max_steps:
#                 break
#     return chain

def reconstruct_chain(
    ranked_blocks: List[Tuple[str, float, BlockRecord]],
    graph: TraceGraph,
    max_steps: int,
    requirement_text: str,
    use_beam: bool = True,
    use_block_graph: bool = True,
    use_diversity: bool = True,
    use_requirement_alignment: bool = True,
    use_dataflow: bool = True,
    transition_mode: str = "static",
    beam_width: int = DEFAULT_BEAM_WIDTH,
    seed_count: int = DEFAULT_BEAM_SEEDS,
    jump_k: int = DEFAULT_BEAM_JUMPS,  # kept for signature compatibility, unused here
    beam_length_alpha: float = 0.65,
    use_transition: bool = True,
) -> List[str]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Build an ordered evidence chain from ranked blocks.
    How to use: Use this to produce a readable trace path after ranking.
    Design note: Beam search is kept strict enough to respect graph structure while still allowing ranked seeds.
    """
    if not ranked_blocks:
        return []

    normalized = sorted(ranked_blocks, key=lambda x: x[1], reverse=True)
    score_map = {bid: score for bid, score, _ in normalized}
    block_map = {bid: block for bid, score, block in normalized}
    ordered_ids = [bid for bid, _, _ in normalized]
    entry_nodes = {bid for bid, _, _ in normalized if graph.block_in_degree(bid) == 0}

    # Non-beam fallback: keep the ranked order (existing behavior).
    # If you want strict graph-constrained behavior even here, this should be changed separately.
    if not use_beam:
        return ordered_ids[:max_steps]

    def beam_value(state) -> float:
        """
        Author: Venkat Kaushal Thippisetty
        Purpose: Beam value.
        How to use: Call this helper as part of the traceability pipeline or supporting utilities.
        Design note: The implementation is documented here to make the codebase easier to read and maintain.
        """
        ln = max(1, len(state["chain"]))
        return state["raw_score"] / math.pow(ln, beam_length_alpha)

    beams = []
    for bid, score, block in normalized[: max(1, seed_count)]:
        bonus = 0.12 if bid in entry_nodes else 0.0
        beams.append(
            {
                "chain": [bid],
                "visited": {bid},
                "raw_score": score + bonus,
                "last": bid,
                "files": {block.file_key},
            }
        )

    for bid in list(entry_nodes)[: max(1, seed_count)]:
        if bid not in {s["last"] for s in beams}:
            beams.append(
                {
                    "chain": [bid],
                    "visited": {bid},
                    "raw_score": score_map.get(bid, 0.0) + 0.12,
                    "last": bid,
                    "files": {block_map[bid].file_key},
                }
            )

    if not beams:
        first = ordered_ids[0]
        beams = [
            {
                "chain": [first],
                "visited": {first},
                "raw_score": score_map.get(first, 0.0),
                "last": first,
                "files": {block_map[first].file_key},
            }
        ]

    best_state = max(beams, key=beam_value)

    for _ in range(1, max_steps):
        expanded = []

        for state in beams:
            last_id = state["last"]
            last_block = block_map[last_id]

            successors: List[str] = []
            if use_block_graph:
                successors = [
                    n
                    for n in graph.block_out_neighbors(last_id)
                    if n in score_map and n not in state["visited"]
                ]

            # Strict graph-constrained behavior:
            # if there is no legal graph successor, this beam cannot continue.
            if not successors:
                continue

            for bid in successors:
                next_block = block_map[bid]

                trans = 0.0
                if use_transition:
                    trans = _transition_bonus(
                        last_block, next_block, graph, transition_mode, use_block_graph
                    )

                file_bonus = 0.03 if next_block.file_key == last_block.file_key else 0.0
                align = (
                    requirement_alignment_score(requirement_text, next_block.text)
                    if use_requirement_alignment
                    else 0.0
                )
                dataflow = dataflow_feature(next_block, graph) if use_dataflow else 0.0

                new_state = {
                    "chain": state["chain"] + [bid],
                    "visited": set(state["visited"]) | {bid},
                    "raw_score": (
                        state["raw_score"]
                        + score_map.get(bid, 0.0)
                        + trans
                        + file_bonus
                        + 0.20 * align
                        + 0.10 * dataflow
                    ),
                    "last": bid,
                    "files": set(state["files"]) | {next_block.file_key},
                }
                expanded.append(new_state)

        if not expanded:
            break

        expanded.sort(key=lambda s: beam_value(s), reverse=True)

        if use_diversity:
            pruned = []
            seen_suffixes = set()
            for s in expanded:
                suffix = tuple(s["chain"][-2:]) if len(s["chain"]) >= 2 else tuple(s["chain"])
                if suffix not in seen_suffixes:
                    pruned.append(s)
                    seen_suffixes.add(suffix)
                if len(pruned) >= beam_width:
                    break
            beams = pruned if pruned else expanded[:beam_width]
        else:
            beams = expanded[:beam_width]

        current_best = max(beams, key=beam_value)
        if beam_value(current_best) > beam_value(best_state):
            best_state = current_best

    best_state = max(beams + [best_state], key=beam_value)
    chain = best_state["chain"][:max_steps]
    return chain

# ============================================================
# Metrics
# ============================================================
def precision(pred_set: Set[str], gt_set: Set[str]) -> float:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Compute set-based precision.
    How to use: Use this for file, block, or evidence retrieval metrics.
    Design note: The helper expects predicted and ground-truth items to be pre-normalized.
    """
    if not pred_set:
        return 0.0
    return len(pred_set & gt_set) / len(pred_set)
def recall(pred_set: Set[str], gt_set: Set[str]) -> float:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Compute set-based recall.
    How to use: Use this for file, block, or evidence retrieval metrics.
    Design note: This keeps the evaluation logic consistent across trace levels.
    """
    if not gt_set:
        return 0.0
    return len(pred_set & gt_set) / len(gt_set)
def chain_order_accuracy(pred_chain: Sequence[str], gt_chain: Sequence[str]) -> float:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Measure how often the predicted order preserves ground-truth pair ordering.
    How to use: Use this when the sequence itself matters, not just membership.
    Design note: A value of 1.0 means every comparable ground-truth pair is ordered correctly.
    """
    if not gt_chain:
        return 0.0
    pos = {item: i for i, item in enumerate(pred_chain)}
    pairs = 0
    correct = 0
    for i in range(len(gt_chain) - 1):
        a, b = gt_chain[i], gt_chain[i + 1]
        if a in pos and b in pos:
            pairs += 1
            if pos[a] < pos[b]:
                correct += 1
    return correct / pairs if pairs else 0.0
def first_correct_position(pred_chain: Sequence[str], gt_chain: Sequence[str]) -> Tuple[int, float]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Return the first predicted position that hits any ground-truth item.
    How to use: Use this as a simple early-hit diagnostic.
    Design note: Lower positions indicate faster evidence discovery.
    """
    gt = set(gt_chain)
    for i, item in enumerate(pred_chain, start=1):
        if item in gt:
            return i, 1.0 / i
    return 0, 0.0
def evidence_coverage(pred_blocks: Sequence[BlockRecord], evidence_items: Sequence[dict]) -> float:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Estimate how much of the cited evidence appears in the predicted blocks.
    How to use: Use this when a requirement includes explicit code snippets as evidence.
    Design note: The check is lexical and conservative by design.
    """
    if not evidence_items:
        return 0.0
    pred_texts = [normalize_text(b.text) for b in pred_blocks]
    covered = 0
    for ev in evidence_items:
        code = ev.get("code", "")
        code_norm = normalize_text(code)
        if not code_norm:
            continue
        code_tokens = tokenize(code_norm)
        matched = False
        for text in pred_texts:
            if code_norm in text:
                matched = True
                break
            if len(code_tokens) >= 3 and overlap_ratio(code_tokens, tokenize(text)) >= 0.45:
                matched = True
                break
        if matched:
            covered += 1
    return covered / len(evidence_items)
def failure_tag(chain_recall: float, order_acc: float, file_precision: float) -> str:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Assign a compact diagnostic label to the current result pattern.
    How to use: Use this to summarize common traceability failure modes.
    Design note: The labels help with quick inspection and ablation analysis.
    """
    if chain_recall >= 0.70 and order_acc <= 0.20:
        return "high_recall_low_order"
    if chain_recall <= 0.35 and file_precision >= 0.60:
        return "good_files_poor_chain"
    if chain_recall >= 0.70 and file_precision <= 0.30:
        return "good_chain_low_precision"
    return "normal"
# ============================================================
# Ablations / run flags
# ============================================================
@dataclass(frozen=True)
class Flags:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Data structure or service for Flags.
    How to use: Instantiate it where the corresponding workflow needs state or behavior.
    Design note: The class groups related responsibilities so the pipeline remains organized.
    """
    use_graph: bool = True
    use_block_graph: bool = True
    use_graph_propagation: bool = True
    use_dataflow: bool = True
    use_structure: bool = True
    use_filename: bool = True
    use_rank_prior: bool = True
    use_alignment: bool = True
    use_diversity: bool = True
    use_beam: bool = True
    use_transition: bool = True
def flags_for_ablation(name: str) -> Flags:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Convert an ablation name into a structured flag set.
    How to use: Use this before running a comparison experiment.
    Design note: Each preset isolates one major subsystem at a time.
    """
    full = Flags()
    if name == "full":
        return full
    if name == "retrieval_only":
        return Flags(
            use_graph=False,
            use_block_graph=False,
            use_graph_propagation=False,
            use_dataflow=False,
            use_structure=False,
            use_filename=False,
            use_rank_prior=False,
            use_alignment=False,
            use_diversity=False,
            use_beam=False,
            use_transition=False,
        )
    if name == "no_graph":
        return Flags(
            use_graph=False,
            use_block_graph=False,
            use_graph_propagation=False,
            use_dataflow=False,
            use_structure=True,
            use_filename=True,
            use_rank_prior=True,
            use_alignment=True,
            use_diversity=True,
            use_beam=True,
            use_transition=True,
        )
    if name == "no_transition":
        return Flags(
            use_graph=True,
            use_block_graph=True,
            use_graph_propagation=True,
            use_dataflow=True,
            use_structure=True,
            use_filename=True,
            use_rank_prior=True,
            use_alignment=True,
            use_diversity=True,
            use_beam=True,
            use_transition=False,
        )
    if name == "no_beam":
        return Flags(
            use_graph=True,
            use_block_graph=True,
            use_graph_propagation=True,
            use_dataflow=True,
            use_structure=True,
            use_filename=True,
            use_rank_prior=True,
            use_alignment=True,
            use_diversity=True,
            use_beam=False,
            use_transition=True,
        )
    if name == "no_dataflow":
        return Flags(
            use_graph=True,
            use_block_graph=True,
            use_graph_propagation=True,
            use_dataflow=False,
            use_structure=True,
            use_filename=True,
            use_rank_prior=True,
            use_alignment=True,
            use_diversity=True,
            use_beam=True,
            use_transition=True,
        )
    if name == "no_alignment":
        return Flags(
            use_graph=True,
            use_block_graph=True,
            use_graph_propagation=True,
            use_dataflow=True,
            use_structure=True,
            use_filename=True,
            use_rank_prior=True,
            use_alignment=False,
            use_diversity=True,
            use_beam=True,
            use_transition=True,
        )
    if name == "no_diversity":
        return Flags(
            use_graph=True,
            use_block_graph=True,
            use_graph_propagation=True,
            use_dataflow=True,
            use_structure=True,
            use_filename=True,
            use_rank_prior=True,
            use_alignment=True,
            use_diversity=False,
            use_beam=True,
            use_transition=True,
        )
    raise ValueError(f"Unknown ablation mode: {name}")
# ============================================================
# Core processing
# ============================================================
def build_indexes(files: Dict[str, str]) -> Tuple[Dict[str, List[BlockRecord]], Dict[str, BlockRecord]]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Build block indexes for the loaded source files.
    How to use: Use this immediately after loading the code corpus.
    Design note: Returns both the file-to-block mapping and the global block lookup table.
    """
    blocks_by_file: Dict[str, List[BlockRecord]] = {}
    block_index: Dict[str, BlockRecord] = {}
    for fk, code in files.items():
        blocks = extract_blocks_from_code(fk, code)
        blocks_by_file[fk] = blocks
        for b in blocks:
            block_index[b.block_id] = b
    return blocks_by_file, block_index
def predicted_files_from_blocks(pred_block_ids: Sequence[str], block_index: Dict[str, BlockRecord]) -> List[str]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Project a chain of blocks back to their parent files.
    How to use: Use this when converting block-level output into file-level traces.
    Design note: The result preserves order and removes duplicates.
    """
    out: List[str] = []
    for bid in pred_block_ids:
        if bid in block_index:
            append_unique(out, block_index[bid].file_key)
    return out
def graph_propagate_block_scores(score_map: Dict[str, float], graph: TraceGraph, candidate_block_ids: Sequence[str], alpha: float = 0.25) -> Dict[str, float]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Smooth block scores through the local graph neighborhood.
    How to use: Use this when neighboring evidence should strengthen one another.
    Design note: Propagation is intentionally gentle so retrieval remains the main signal.
    """
    out = dict(score_map)
    candidate_set = set(candidate_block_ids)
    for bid in candidate_block_ids:
        neighbors = [n for n in graph.block_neighbors(bid) if n in candidate_set]
        if not neighbors:
            continue
        neighbor_mean = statistics.mean(score_map.get(n, 0.0) for n in neighbors)
        out[bid] = (1 - alpha) * score_map.get(bid, 0.0) + alpha * neighbor_mean
    return out
def select_top_blocks(candidates: List[Tuple[str, float, BlockRecord, Dict[str, float]]], k: int, use_diversity: bool) -> List[str]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Select the strongest blocks while penalizing redundancy.
    How to use: Use this after ranking to create a compact evidence set.
    Design note: The MMR-style trade-off keeps the result concise but diverse.
    """
    if not candidates:
        return []
    selected: List[str] = []
    selected_texts: List[str] = []
    remaining = candidates[:]
    while remaining and len(selected) < k:
        best_idx = None
        best_val = None
        for i, (bid, score, block, feats) in enumerate(remaining):
            div = diversity_penalty(selected_texts, block.text) if use_diversity else 0.0
            mmr = MMR_LAMBDA * score - (1.0 - MMR_LAMBDA) * div
            if best_val is None or mmr > best_val:
                best_val = mmr
                best_idx = i
        bid, score, block, feats = remaining.pop(best_idx)  # type: ignore[arg-type]
        selected.append(bid)
        selected_texts.append(block.text)
    return selected
def process_requirement(
    req: dict,
    files: Dict[str, str],
    file_suite: RetrievalSuite,
    block_suite: RetrievalSuite,
    graph: TraceGraph,
    spec: RunSpec,
    flags: Flags,
    top_k_files: int,
    top_k_blocks: int,
    beam_length_alpha: float,
) -> RequirementReport:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Run the full retrieval, ranking, chain, and evaluation pipeline for one requirement.
    How to use: Use this as the core per-item experiment step.
    Design note: This is the central orchestration point of the script.
    """
    requirement_id = req.get("id", req.get("requirement_id", ""))
    requirement = req.get("requirement", req.get("original", req.get("query", ""))) or ""
    paraphrases = req.get("paraphrases", []) or []
    gt_chain = normalize_ground_truth_chain(req.get("ground_truth", {}).get("chain", []) or [])
    gt_evidence = req.get("ground_truth", {}).get("evidence", []) or []
    gt_files = gt_files_from_chain(gt_chain)
    gt_blocks = gt_block_ids_from_chain(gt_chain)
    query_specs = build_query_specs(req, spec.query_agg)
    file_scores, file_debug = file_suite.score(query_specs, spec.query_agg, spec.retriever_mode)
    block_scores, block_debug = block_suite.score(query_specs, spec.query_agg, spec.retriever_mode)
    file_order = list(np.argsort(-file_scores))
    raw_retrieved = [file_suite.item_keys[i] for i in file_order[:top_k_files]]
    candidate_file_keys = list(raw_retrieved)
    if flags.use_graph:
        for fk in list(candidate_file_keys):
            for nb in graph.file_neighbors(fk):
                append_unique(candidate_file_keys, nb)
    candidate_blocks: List[BlockRecord] = []
    block_records: List[Tuple[str, float, BlockRecord, Dict[str, float]]] = []
    file_rank_map = {fk: i + 1 for i, fk in enumerate(file_order_to_keys(file_order, file_suite.item_keys))}
    for fk in candidate_file_keys:
        for b in graph.blocks_by_file.get(fk, []):
            candidate_blocks.append(b)
    if not candidate_blocks:
        empty_metrics = {
            "raw_file_precision@k": 0.0,
            "raw_file_recall@k": 0.0,
            "reranked_file_precision@k": 0.0,
            "reranked_file_recall@k": 0.0,
            "strict_file_precision": 0.0,
            "strict_file_recall": 0.0,
            "file_precision": 0.0,
            "file_recall": 0.0,
            "block_chain_recall": 0.0,
            "block_lcs_recall": 0.0,
            "block_order_accuracy": 0.0,
            "evidence_coverage": 0.0,
            "chain_length_ratio": 0.0,
            "first_correct_position": 0.0,
            "first_correct_rr": 0.0,
        }
        return RequirementReport(
            requirement_id=requirement_id,
            requirement=requirement,
            paraphrases=paraphrases,
            ground_truth_chain=gt_chain,
            ground_truth_evidence=gt_evidence,
            retrieval={"raw_files": [], "candidate_file_keys": []},
            block_ranking={"top_blocks": [], "all_blocks": []},
            chain_result={"predicted_chain": [], "predicted_files": [], "strict_selected": [], "conservative_selected": []},
            metrics=empty_metrics,
            debug={"file_debug": file_debug, "block_debug": block_debug},
        )
    # Build per-block features.
    for b in candidate_blocks:
        b_idx = block_suite.item_keys.index(b.block_id)
        fk_idx = file_suite.item_keys.index(b.file_key)
        block_retrieval = float(block_scores[b_idx])
        file_retrieval = float(file_scores[fk_idx])
        feats = {
            "retrieval": block_retrieval,
            "file_prior": file_retrieval,
            "filename": filename_feature(requirement, b.file_key) if flags.use_filename else 0.0,
            "rank_prior": retrieval_rank_feature(file_rank_map.get(b.file_key, top_k_files + 1), max(1, len(file_suite.item_keys))) if flags.use_rank_prior else 0.0,
            "structure": block_structure_feature(b) if flags.use_structure else 0.0,
            "graph_support": min(1.0, len(graph.block_neighbors(b.block_id)) / 6.0) if flags.use_graph else 0.0,
            "dataflow": dataflow_feature(b, graph) if flags.use_dataflow else 0.0,
            "alignment": requirement_alignment_score(requirement, b.text) if flags.use_alignment else 0.0,
        }
        block_records.append((b.block_id, combine_block_features(feats), b, feats))
    # Normalize each feature across the candidate set, then rescore.
    feature_keys = list(block_records[0][3].keys())
    norm_rows: List[Dict[str, float]] = []
    for key in feature_keys:
        vals = [row[3].get(key, 0.0) for row in block_records]
        nvals = minmax_norm(vals)
        for i, nv in enumerate(nvals):
            if i >= len(norm_rows):
                norm_rows.append({})
            norm_rows[i][key] = float(nv)
    rescored_blocks: List[Tuple[str, float, BlockRecord, Dict[str, float]]] = []
    for i, (bid, _, blk, feats) in enumerate(block_records):
        nfeats = dict(norm_rows[i])
        # keep retrieval slightly dominant; this is still a traceability system, not just a ranking model
        score = combine_block_features(nfeats)
        rescored_blocks.append((bid, score, blk, nfeats))
    # Optional graph propagation
    if flags.use_graph and flags.use_graph_propagation:
        score_map = {bid: sc for bid, sc, _, _ in rescored_blocks}
        block_ids = [bid for bid, _, _, _ in rescored_blocks]
        score_map = graph_propagate_block_scores(score_map, graph, block_ids, alpha=0.25)
        rescored_blocks = [(bid, score_map.get(bid, 0.0), blk, feats) for bid, _, blk, feats in rescored_blocks]
    rescored_blocks.sort(key=lambda x: x[1], reverse=True)
    scores = [s for _, s, _, _ in rescored_blocks]
    max_score = max(scores) if scores else 0.0
    mean_score = statistics.mean(scores) if scores else 0.0
    stdev_score = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    strict_threshold = max(max_score * STRICT_RATIO, mean_score + 0.10 * stdev_score)
    conservative_threshold = max(max_score * CONSERVATIVE_RATIO, mean_score - 0.10 * stdev_score)
    strict_pool = [(bid, sc, blk, feats) for bid, sc, blk, feats in rescored_blocks if sc >= strict_threshold]
    if not strict_pool:
        strict_pool = [rescored_blocks[0]]
    strict_selected = select_top_blocks(strict_pool, k=min(top_k_blocks, len(strict_pool)), use_diversity=flags.use_diversity)
    conservative_pool = [(bid, sc, blk, feats) for bid, sc, blk, feats in rescored_blocks if sc >= conservative_threshold]
    if not conservative_pool:
        conservative_pool = [rescored_blocks[0]]
    conservative_selected = select_top_blocks(conservative_pool, k=min(top_k_blocks, len(conservative_pool)), use_diversity=flags.use_diversity)
    ranked_blocks = [(bid, sc, blk) for bid, sc, blk, _ in rescored_blocks]
    predicted_chain = reconstruct_chain(
        ranked_blocks=ranked_blocks,
        graph=graph,
        max_steps=max(len(gt_chain), 3),
        requirement_text=requirement,
        use_beam=flags.use_beam,
        use_block_graph=flags.use_block_graph,
        use_diversity=flags.use_diversity,
        use_requirement_alignment=flags.use_alignment,
        use_dataflow=flags.use_dataflow,
        transition_mode=spec.transition_mode,
        beam_width=DEFAULT_BEAM_WIDTH,
        seed_count=DEFAULT_BEAM_SEEDS,
        jump_k=DEFAULT_BEAM_JUMPS,
        beam_length_alpha=beam_length_alpha,
        use_transition=flags.use_transition,
    )
    for bid in strict_selected:
        append_unique(predicted_chain, bid)
    predicted_blocks = [graph.block_index[bid] for bid in predicted_chain if bid in graph.block_index]
    predicted_files = predicted_files_from_blocks(predicted_chain, graph.block_index)
    gt_block_set = set(gt_blocks)
    pred_block_set = set(predicted_chain)
    block_chain_recall = len(gt_block_set & pred_block_set) / len(gt_block_set) if gt_block_set else 0.0
    pred_lcs = lcs_length(predicted_chain, gt_blocks)
    block_lcs_recall = pred_lcs / len(gt_blocks) if gt_blocks else 0.0
    order_acc = chain_order_accuracy(predicted_chain, gt_blocks)
    ev_cov = evidence_coverage(predicted_blocks, gt_evidence)
    first_pos, first_rr = first_correct_position(predicted_chain, gt_blocks)
    chain_length_ratio = len(predicted_chain) / max(1, len(gt_chain))
    gt_file_set = {normalize_compact(f) for f in gt_files}
    pred_file_set = {normalize_compact(f) for f in predicted_files}
    raw_file_set = {normalize_compact(file_suite.item_keys[i]) for i in file_order[:top_k_files]}
    reranked_file_set = {normalize_compact(file_suite.item_keys[i]) for i in np.argsort(-file_scores)[:top_k_files]}
    strict_pred_file_set = {normalize_compact(f) for f in predicted_files_from_blocks(strict_selected, graph.block_index)}
    metrics = {
        "raw_file_precision@k": precision(raw_file_set, gt_file_set),
        "raw_file_recall@k": recall(raw_file_set, gt_file_set),
        "reranked_file_precision@k": precision(reranked_file_set, gt_file_set),
        "reranked_file_recall@k": recall(reranked_file_set, gt_file_set),
        "strict_file_precision": precision(strict_pred_file_set, gt_file_set),
        "strict_file_recall": recall(strict_pred_file_set, gt_file_set),
        "file_precision": precision(pred_file_set, gt_file_set),
        "file_recall": recall(pred_file_set, gt_file_set),
        "block_chain_recall": block_chain_recall,
        "block_lcs_recall": block_lcs_recall,
        "block_order_accuracy": order_acc,
        "evidence_coverage": ev_cov,
        "chain_length_ratio": chain_length_ratio,
        "first_correct_position": float(first_pos),
        "first_correct_rr": float(first_rr),
    }
    raw_files = [{"rank": i + 1, "file_key": file_suite.item_keys[idx], "score": round(float(file_scores[idx]), 4)} for i, idx in enumerate(file_order[:top_k_files])]
    top_blocks = [{"rank": i + 1, "block_id": bid, "file_key": blk.file_key, "label": blk.label, "score": round(float(sc), 4)} for i, (bid, sc, blk, _) in enumerate(rescored_blocks[:top_k_blocks])]
    return RequirementReport(
        requirement_id=requirement_id,
        requirement=requirement,
        paraphrases=paraphrases,
        ground_truth_chain=gt_chain,
        ground_truth_evidence=gt_evidence,
        retrieval={"raw_files": raw_files, "candidate_file_keys": candidate_file_keys},
        block_ranking={"top_blocks": top_blocks, "all_blocks": []},
        chain_result={
            "predicted_chain": predicted_chain,
            "predicted_files": predicted_files,
            "strict_selected": strict_selected,
            "conservative_selected": conservative_selected,
        },
        metrics=metrics,
        debug={
            "file_debug": file_debug,
            "block_debug": block_debug,
            "strict_threshold": strict_threshold,
            "conservative_threshold": conservative_threshold,
            "candidate_file_keys": candidate_file_keys,
            "spec": dataclasses.asdict(spec),
            "flags": dataclasses.asdict(flags),
        },
    )
def file_order_to_keys(order: Sequence[int], keys: Sequence[str]) -> List[str]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: File order to keys.
    How to use: Call this helper as part of the traceability pipeline or supporting utilities.
    Design note: The implementation is documented here to make the codebase easier to read and maintain.
    """
    return [keys[i] for i in order]
# ============================================================
# Experiment runners
# ============================================================
def summarize_reports(reports: List[RequirementReport]) -> Dict[str, float]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Aggregate per-requirement reports into summary statistics.
    How to use: Use this after a full experiment run completes.
    Design note: The function prepares both averages and diagnostics for comparison.
    """
    if not reports:
        return {}
    metrics_keys = list(reports[0].metrics.keys())
    summary: Dict[str, float] = {}
    for key in metrics_keys:
        vals = [float(r.metrics.get(key, 0.0)) for r in reports]
        summary[key] = float(np.mean(vals)) if vals else 0.0
    return summary
def run_spec(
    requirements: List[dict],
    files: Dict[str, str],
    file_suite: RetrievalSuite,
    block_suite: RetrievalSuite,
    graph: TraceGraph,
    spec: RunSpec,
    flags: Flags,
    top_k_files: int,
    top_k_blocks: int,
    beam_length_alpha: float,
    save_failures: bool,
    failure_chain_recall: float,
    failure_order_accuracy: float,
    save_requirement_logs: bool,
    requirement_log_root: str,
) -> Dict[str, Any]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Run a single experiment configuration over the dataset.
    How to use: Use this for one retriever, one aggregation strategy, and one ablation setting.
    Design note: This is the batch-level driver that repeatedly calls the per-requirement pipeline.
    """
    reports: List[RequirementReport] = []
    failure_rows: List[Dict[str, Any]] = []
    spec_dir = os.path.join(requirement_log_root, safe_filename(f"{spec.retriever_mode}__{spec.query_agg}__{spec.ablation_mode}__{spec.transition_mode}", fallback="spec"))
    if save_requirement_logs:
        ensure_dir(spec_dir)
    for req_idx, req in enumerate(tqdm(requirements, desc=f"{spec.retriever_mode}/{spec.query_agg}/{spec.ablation_mode}/{spec.transition_mode}"), start=1):
        rep = process_requirement(
            req=req,
            files=files,
            file_suite=file_suite,
            block_suite=block_suite,
            graph=graph,
            spec=spec,
            flags=flags,
            top_k_files=top_k_files,
            top_k_blocks=top_k_blocks,
            beam_length_alpha=beam_length_alpha,
        )
        reports.append(rep)
        if save_requirement_logs:
            payload = report_to_dict(rep, spec=spec, index=req_idx)
            payload["retriever_mode"] = spec.retriever_mode
            payload["query_agg"] = spec.query_agg
            payload["ablation_mode"] = spec.ablation_mode
            payload["transition_mode"] = spec.transition_mode
            req_id = safe_filename(rep.requirement_id or f"REQ_{req_idx:04d}", fallback=f"REQ_{req_idx:04d}")
            write_json(os.path.join(spec_dir, f"{req_idx:04d}_{req_id}.json"), payload)
        if save_failures:
            cr = float(rep.metrics.get("block_chain_recall", 0.0))
            oa = float(rep.metrics.get("block_order_accuracy", 0.0))
            if cr >= failure_chain_recall and oa <= failure_order_accuracy:
                failure_rows.append({
                    "requirement_id": rep.requirement_id,
                    "requirement": rep.requirement,
                    "retriever_mode": spec.retriever_mode,
                    "query_agg": spec.query_agg,
                    "ablation_mode": spec.ablation_mode,
                    "transition_mode": spec.transition_mode,
                    "chain_recall": cr,
                    "order_accuracy": oa,
                    "file_precision": float(rep.metrics.get("file_precision", 0.0)),
                    "file_recall": float(rep.metrics.get("file_recall", 0.0)),
                    "first_correct_position": float(rep.metrics.get("first_correct_position", 0.0)),
                    "predicted_chain": rep.chain_result.get("predicted_chain", []),
                    "ground_truth_chain": rep.ground_truth_chain,
                })
    if save_requirement_logs:
        write_requirement_logs(reports, spec, spec_dir)
    summary = summarize_reports(reports)
    per_req_rows = []
    for r in reports:
        row = {
            "requirement_id": r.requirement_id,
            "requirement": r.requirement,
            "retriever_mode": spec.retriever_mode,
            "query_agg": spec.query_agg,
            "ablation_mode": spec.ablation_mode,
            "transition_mode": spec.transition_mode,
        }
        row.update(r.metrics)
        per_req_rows.append(row)
    return {
        "spec": dataclasses.asdict(spec),
        "flags": dataclasses.asdict(flags),
        "summary": summary,
        "reports": reports,
        "per_requirement_rows": per_req_rows,
        "failure_rows": failure_rows,
    }
def flatten_summary_rows(all_outputs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Convert a nested summary object into CSV-ready rows.
    How to use: Use this before exporting experiment results to spreadsheets.
    Design note: This keeps the final export simple and machine-friendly.
    """
    rows = []
    for spec_name, payload in all_outputs.items():
        s = payload["summary"]
        spec = payload["spec"]
        rows.append({
            "spec": spec_name,
            "retriever_mode": spec["retriever_mode"],
            "query_agg": spec["query_agg"],
            "ablation_mode": spec["ablation_mode"],
            "transition_mode": spec["transition_mode"],
            "raw_file_precision@k": s.get("raw_file_precision@k", 0.0),
            "raw_file_recall@k": s.get("raw_file_recall@k", 0.0),
            "reranked_file_precision@k": s.get("reranked_file_precision@k", 0.0),
            "reranked_file_recall@k": s.get("reranked_file_recall@k", 0.0),
            "strict_file_precision": s.get("strict_file_precision", 0.0),
            "strict_file_recall": s.get("strict_file_recall", 0.0),
            "file_precision": s.get("file_precision", 0.0),
            "file_recall": s.get("file_recall", 0.0),
            "block_chain_recall": s.get("block_chain_recall", 0.0),
            "block_lcs_recall": s.get("block_lcs_recall", 0.0),
            "block_order_accuracy": s.get("block_order_accuracy", 0.0),
            "evidence_coverage": s.get("evidence_coverage", 0.0),
            "chain_length_ratio": s.get("chain_length_ratio", 0.0),
            "first_correct_position": s.get("first_correct_position", 0.0),
            "first_correct_rr": s.get("first_correct_rr", 0.0),
        })
    return rows
def save_rows_csv(path: str, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Write summary rows to a CSV file.
    How to use: Use this for human review or post-processing in spreadsheets.
    Design note: The helper accepts any row set with consistent columns.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
def compare_full_vs_baseline(all_outputs: Dict[str, Dict[str, Any]], baseline_ablation: str = "retrieval_only") -> List[Dict[str, Any]]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Compare the full system against a baseline configuration.
    How to use: Use this when you want a direct ablation-style difference report.
    Design note: The comparison emphasizes both central tendency and bootstrap confidence intervals.
    """
    out: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for spec_name, payload in all_outputs.items():
        spec = payload["spec"]
        key = (spec["retriever_mode"], spec["query_agg"], spec["transition_mode"])
        by_key[(spec["retriever_mode"], spec["query_agg"], spec["transition_mode"])] = payload
    for (retriever_mode, query_agg, transition_mode), payload in by_key.items():
        full_key = None
        base_key = None
        for spec_name, p in all_outputs.items():
            s = p["spec"]
            if s["retriever_mode"] == retriever_mode and s["query_agg"] == query_agg and s["transition_mode"] == transition_mode:
                if s["ablation_mode"] == "full":
                    full_key = spec_name
                if s["ablation_mode"] == baseline_ablation:
                    base_key = spec_name
        if full_key is None or base_key is None:
            continue
        full_rows = all_outputs[full_key]["per_requirement_rows"]
        base_rows = all_outputs[base_key]["per_requirement_rows"]
        if len(full_rows) != len(base_rows):
            continue
        metrics = ["block_chain_recall", "block_lcs_recall", "block_order_accuracy", "evidence_coverage", "file_precision", "file_recall"]
        for metric in metrics:
            a = [float(r.get(metric, 0.0)) for r in full_rows]
            b = [float(r.get(metric, 0.0)) for r in base_rows]
            stats = bootstrap_mean_diff(a, b)
            stats.update({"retriever_mode": retriever_mode, "query_agg": query_agg, "transition_mode": transition_mode, "metric": metric})
            out.append(stats)
    return out
def plot_summary(rows: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Generate comparison plots for the experiment summaries.
    How to use: Use this when visual inspection is useful alongside numeric tables.
    Design note: Plots are saved to disk instead of being shown interactively.
    """
    ensure_dir(output_dir)
    if not rows:
        return
    metrics = ["block_chain_recall", "block_order_accuracy", "evidence_coverage", "file_recall"]
    for metric in metrics:
        fig = plt.figure(figsize=(max(10, len(rows) * 0.9), 5))
        labels = [r["spec"] for r in rows]
        vals = [r[metric] for r in rows]
        x = np.arange(len(labels))
        plt.bar(x, vals)
        plt.xticks(x, labels, rotation=35, ha="right")
        plt.ylabel(metric)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{metric}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
def parse_csv_list(text: str, default: List[str]) -> List[str]:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Parse a comma-separated command-line list into trimmed items.
    How to use: Use this for CLI arguments that allow multiple modes or datasets.
    Design note: Empty entries are removed so argument parsing stays forgiving.
    """
    if not text:
        return default
    items = [x.strip().lower() for x in text.split(",") if x.strip()]
    return items or default
# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Define and parse the command-line interface.
    How to use: Use this as the program entry point for configuration.
    Design note: The CLI centralizes all tunable experiment settings in one place.
    """
    p = argparse.ArgumentParser(description="Modern COBOL traceability experiment runner.")
    p.add_argument("--dataset", required=True, help="Path to dataset JSON")
    p.add_argument("--files", required=True, help="Path to files JSON")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    p.add_argument("--retrievers", default=",".join(DEFAULT_RETRIEVERS), help="Comma-separated list: tfidf,lsa,bm25,embed,hybrid_fixed,hybrid_adaptive")
    p.add_argument("--query_aggs", default=",".join(DEFAULT_QUERY_AGGS), help="Comma-separated list: max,weighted,softmax")
    p.add_argument("--ablation_modes", default=",".join(DEFAULT_ABLATIONS), help="Comma-separated list of ablations")
    p.add_argument("--transition_modes", default=DEFAULT_TRANSITION_MODE, help="Comma-separated list: static,adaptive")
    p.add_argument("--top_k_files", type=int, default=DEFAULT_TOP_K_FILES)
    p.add_argument("--top_k_blocks", type=int, default=DEFAULT_TOP_K_BLOCKS)
    p.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--beam_length_alpha", type=float, default=0.65)
    p.add_argument("--save_per_requirement", action="store_true")
    p.add_argument("--save_requirement_logs", dest="save_requirement_logs", action="store_true")
    p.add_argument("--no_save_requirement_logs", dest="save_requirement_logs", action="store_false")
    p.set_defaults(save_requirement_logs=True)
    p.add_argument("--requirement_log_dir", default="requirement_logs")
    p.add_argument("--save_failures", action="store_true")
    p.add_argument("--failure_chain_recall", type=float, default=0.70)
    p.add_argument("--failure_order_accuracy", type=float, default=0.20)
    p.add_argument("--plot", action="store_true")
    return p.parse_args()
# ============================================================
# Main
# ============================================================
def main() -> int:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Execute the full experiment workflow.
    How to use: Use this when the script is run as a standalone program.
    Design note: It ties together parsing, loading, execution, reporting, and plotting.
    """
    args = parse_args()
    ensure_dir(args.output_dir)
    requirements = load_dataset_json(args.dataset)
    files = load_files_json(args.files)
    blocks_by_file, block_index = build_indexes(files)
    graph = TraceGraph(files, blocks_by_file)
    file_corpus = [build_file_view(fk, files[fk]) for fk in files]
    file_ids = list(files.keys())
    block_ids = []
    block_corpus = []
    for fk, blocks in blocks_by_file.items():
        for b in blocks:
            block_ids.append(b.block_id)
            block_corpus.append(b.text)
    file_suite = RetrievalSuite(file_corpus, file_ids, args.embedding_model)
    block_suite = RetrievalSuite(block_corpus, block_ids, args.embedding_model)
    retrievers = parse_csv_list(args.retrievers, DEFAULT_RETRIEVERS)
    query_aggs = parse_csv_list(args.query_aggs, DEFAULT_QUERY_AGGS)
    ablation_modes = parse_csv_list(args.ablation_modes, DEFAULT_ABLATIONS)
    transition_modes = parse_csv_list(args.transition_modes, [DEFAULT_TRANSITION_MODE])
    for r in retrievers:
        if r not in DEFAULT_RETRIEVERS:
            raise ValueError(f"Unknown retriever mode: {r}")
    for q in query_aggs:
        if q not in {"max", "weighted", "softmax"}:
            raise ValueError(f"Unknown query aggregation mode: {q}")
    for a in ablation_modes:
        if a not in DEFAULT_ABLATIONS:
            raise ValueError(f"Unknown ablation mode: {a}")
    for t in transition_modes:
        if t not in {"static", "adaptive"}:
            raise ValueError(f"Unknown transition mode: {t}")
    all_outputs: Dict[str, Dict[str, Any]] = {}
    for retriever_mode in retrievers:
        for query_agg in query_aggs:
            for ablation_mode in ablation_modes:
                for transition_mode in transition_modes:
                    spec = RunSpec(
                        retriever_mode=retriever_mode,
                        query_agg=query_agg,
                        ablation_mode=ablation_mode,
                        transition_mode=transition_mode,
                    )
                    flags = flags_for_ablation(ablation_mode)
                    key = f"{retriever_mode}__{query_agg}__{ablation_mode}__{transition_mode}"
                    all_outputs[key] = run_spec(
                        requirements=requirements,
                        files=files,
                        file_suite=file_suite,
                        block_suite=block_suite,
                        graph=graph,
                        spec=spec,
                        flags=flags,
                        top_k_files=args.top_k_files,
                        top_k_blocks=args.top_k_blocks,
                        beam_length_alpha=args.beam_length_alpha,
                        save_failures=args.save_failures,
                        failure_chain_recall=args.failure_chain_recall,
                        failure_order_accuracy=args.failure_order_accuracy,
                        save_requirement_logs=args.save_requirement_logs,
                        requirement_log_root=os.path.join(args.output_dir, args.requirement_log_dir),
                    )
    summary_rows = flatten_summary_rows(all_outputs)
    print("\n=== Retrieval / Structure / Ordering summary ===")
    print_table(
        summary_rows,
        [
            "spec",
            "raw_file_precision@k",
            "raw_file_recall@k",
            "file_precision",
            "file_recall",
            "block_chain_recall",
            "block_lcs_recall",
            "block_order_accuracy",
            "evidence_coverage",
            "first_correct_position",
        ],
    )
    write_json(os.path.join(args.output_dir, "graph_report.json"), graph.report)
    write_json(os.path.join(args.output_dir, "summary.json"), {k: v["summary"] for k, v in all_outputs.items()})
    write_json(os.path.join(args.output_dir, "manifest.json"), {
        "dataset": args.dataset,
        "files": args.files,
        "retrievers": retrievers,
        "query_aggs": query_aggs,
        "ablation_modes": ablation_modes,
        "transition_modes": transition_modes,
        "top_k_files": args.top_k_files,
        "top_k_blocks": args.top_k_blocks,
        "embedding_model": args.embedding_model,
        "beam_length_alpha": args.beam_length_alpha,
        "requirements": len(requirements),
        "files_count": len(files),
        "blocks_count": len(block_index),
    })
    save_rows_csv(
        os.path.join(args.output_dir, "retriever_comparison.csv"),
        summary_rows,
        [
            "spec",
            "retriever_mode",
            "query_agg",
            "ablation_mode",
            "transition_mode",
            "raw_file_precision@k",
            "raw_file_recall@k",
            "reranked_file_precision@k",
            "reranked_file_recall@k",
            "strict_file_precision",
            "strict_file_recall",
            "file_precision",
            "file_recall",
            "block_chain_recall",
            "block_lcs_recall",
            "block_order_accuracy",
            "evidence_coverage",
            "chain_length_ratio",
            "first_correct_position",
            "first_correct_rr",
        ],
    )
    if args.save_per_requirement:
        for spec_name, payload in all_outputs.items():
            rows = payload["per_requirement_rows"]
            if rows:
                out_csv = os.path.join(args.output_dir, f"per_requirement_{spec_name}.csv")
                save_rows_csv(out_csv, rows, list(rows[0].keys()))
    if args.save_failures:
        failure_path = os.path.join(args.output_dir, "failure_cases.jsonl")
        with open(failure_path, "w", encoding="utf-8") as f:
            for spec_name, payload in all_outputs.items():
                for row in payload.get("failure_rows", []):
                    row = dict(row)
                    row["spec"] = spec_name
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    compare_rows = compare_full_vs_baseline(all_outputs, baseline_ablation="retrieval_only")
    write_json(os.path.join(args.output_dir, "bootstrap_comparison.json"), compare_rows)
    write_json(os.path.join(args.output_dir, "detailed_summary.json"), {
        "summary_rows": summary_rows,
        "bootstrap_comparison": compare_rows,
        "all_outputs_summary": {k: v["summary"] for k, v in all_outputs.items()},
    })
    print("\n=== Bootstrap comparison (full - retrieval_only) ===")
    if compare_rows:
        print_table(compare_rows, ["retriever_mode", "query_agg", "transition_mode", "metric", "mean_diff", "ci_low", "ci_high", "p_value"])
    else:
        print("No paired comparisons available.")
    if args.plot:
        plot_summary(summary_rows, args.output_dir)
    print(f"\nSaved outputs to: {args.output_dir}")
    return 0
def print_table(rows: List[Dict[str, Any]], columns: List[str]) -> None:
    """
    Author: Venkat Kaushal Thippisetty
    Purpose: Print a compact aligned text table.
    How to use: Use this for console summaries when CSV files are not enough.
    Design note: This is intentionally lightweight and dependency-free.
    """
    if not rows:
        print("No rows to display.")
        return
    if pd is not None:
        df = pd.DataFrame(rows)[columns]
        print(df.to_string(index=False))
        return
    widths = {c: max(len(c), max(len(f"{r.get(c, ''):.4f}") if isinstance(r.get(c), float) else len(str(r.get(c, ""))) for r in rows)) for c in columns}
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    sep = "-+-".join("-" * widths[c] for c in columns)
    print(header)
    print(sep)
    for r in rows:
        parts = []
        for c in columns:
            v = r.get(c, "")
            txt = f"{v:.4f}" if isinstance(v, float) else str(v)
            parts.append(txt.ljust(widths[c]))
        print(" | ".join(parts))
if __name__ == "__main__":
    raise SystemExit(main())
