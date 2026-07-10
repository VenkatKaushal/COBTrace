# COBTracer

COBTracer is a framework for reconstructing execution traces from natural-language requirements in legacy COBOL systems.

This repository accompanies the paper:

**COBTracer: An Empirical Study of Execution Trace Reconstruction in Legacy COBOL Systems**

## Overview

COBTracer combines:

- multi-retrieval evidence generation
- evidence fusion
- graph propagation
- execution trace reconstruction

It supports multiple retrieval strategies, query aggregation modes, ablation settings, and statistical analysis.

## Requirements

- Python 3.10+
- `numpy`
- `scikit-learn`
- `matplotlib`
- `tqdm`
- `pandas`
- `rank-bm25`
- `sentence-transformers`

## Installation

```bash
pip install -r requirements.txt
```

## Input Format

### Dataset file

Pass a JSON file containing requirements.

Supported formats:

```json
[
  {
    "id": "REQ-1",
    "requirement": "The system shall ...",
    "ground_truth": {
      "chain": [
        {"file": "FILE1.cbl", "block": "BLOCK1", "role": "entry"}
      ],
      "evidence": [
        {"code": "MOVE ..."}
      ]
    }
  }
]
```

or:

```json
{
  "requirements": [
    {
      "id": "REQ-1",
      "requirement": "The system shall ..."
    }
  ]
}
```

### Files file

Pass a JSON file containing COBOL source files and their contents.

Supported formats:

```json
{
  "FILE1.cbl": "COBOL source code here",
  "FILE2.cbl": "More source code here"
}
```

or:

```json
{
  "files": [
    {"file": "FILE1.cbl", "code": "COBOL source code here"},
    {"file": "FILE2.cbl", "code": "More source code here"}
  ]
}
```

## Running

```bash
python Experiment.py --dataset path/to/requirements.json --files path/to/files.json
```

Common options:

```bash
python Experiment.py   --dataset path/to/requirements.json   --files path/to/files.json   --output_dir outputs   --retrievers tfidf,lsa,bm25,embed,hybrid_fixed,hybrid_adaptive   --query_aggs weighted   --ablation_modes full,retrieval_only,no_graph,no_transition,no_beam   --transition_modes static,adaptive   --top_k_files 5   --top_k_blocks 8   --plot
```

Use `--help` to see all available options.

## Retrievers

- `tfidf`
- `lsa`
- `bm25`
- `embed`
- `hybrid_fixed`
- `hybrid_adaptive`

## Query Aggregation

- `max`
- `weighted`
- `softmax`

## Ablations

- `full`
- `retrieval_only`
- `no_graph`
- `no_transition`
- `no_beam`

## Outputs

The script writes results to the output directory.

Typical outputs include:

- experiment summaries
- per-requirement JSON reports
- JSONL logs
- requirement-level summaries
- failure cases, if enabled
- plots, if enabled

## Output Directory Structure

A typical run may create:

```text
outputs/
├── requirements_summary.json
├── requirements.jsonl
├── <spec>/
│   ├── 0001_REQ-1.json
│   ├── 0002_REQ-2.json
│   └── ...
└── plots/
    ├── block_chain_recall.png
    ├── block_order_accuracy.png
    ├── evidence_coverage.png
    └── file_recall.png
```

## Citation

If you use COBTracer, please cite the paper.


## Authors

- Venkat Kaushal Thippisetty
- Sridhar Chimalakonda
Affiliation: Indian Institute of Technology Tirupati

## License
