# Retrieval-Augmented Generation (RAG) Model Evaluation and Benchmarking

This repository hosts code, data, and evaluation metrics for benchmarking different Retrieval-Augmented Generation (RAG) approaches on nuclear policy and regulatory documents. The project compares baseline and graph-based RAG models across multiple QA benchmarks and visualizes performance using radar plots.

## Table of Contents

* [Overview](#overview)
* [Repository Structure](#repository-structure)
* [Installation](#installation)
* [Data](#data)
* [Usage](#usage)

  * [Running Models](#running-models)
  * [Evaluation](#evaluation)
  * [Visualizations](#visualizations)
* [Benchmarks](#benchmarks)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)

## Overview

This project evaluates two primary RAG configurations:

1. **Baseline RAG**: A standard retrieval + generation pipeline (`base_model.py`, `metrics_basemodel.py`).
2. **Graph RAG (G-RAG)**: A graph-augmented retrieval pipeline (`grag.py`, `metrics.py`).

The goal is to determine which approach yields better performance on QA tasks drawn from nuclear policy documents.

## Repository Structure

```
├── benchmarks/                # Benchmark repositories (ALCE, CRAG, EXPERT2, Retrieval-QA-Benchmark, ScienceQA)
├── data/                      # Source documents (IAEA & NRC policy PDFs)
│   ├── Accident Management Programmes for Nuclear....pdf
│   ├── Advisory Material for the IAEA Regulations for t....pdf
│   ├── Ageing Management for Research Reactors.pdf
│   ├── Application of the Concept of Clearance.pdf
│   └── ...
├── base_model.py              # Baseline RAG implementation
├── grag.py                    # Graph-augmented RAG implementation
├── model.py                   # Shared model utilities
├── odo_large_test.py          # Large-scale RAG tests
├── odotets.py                 # ODoT evaluation scripts
├── requirements.txt           # Python dependencies
├── metrics_basemodel.py       # Metrics for baseline model
├── metrics.py                 # Metrics for graph RAG
├── radar_plot.py              # Radar plot visualization script
├── model_comparison_radar_plot.png  # Generated comparison plot
├── model_evaluation_radar_plot.png  # Generated evaluation plot
└── README.md                  # Project README
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<username>/dse697_LU_NS.git
   cd dse697_LU_NS
   ```
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

## Data

* The `/data` directory contains IAEA and NRC policy documents in PDF format.
* The `/benchmarks` folder includes QA benchmark repositories used for evaluation (ALCE, CRAG, EXPERT2, Retrieval-QA-Benchmark, ScienceQA).
* Example evaluation outputs (`RAG_basemodel.csv`, `RAG_evaluation_results.csv`, etc.) are stored in the root.

## Usage

### Running Models

* **Baseline RAG**:

  ```bash
  python base_model.py --data_dir data/ --benchmarks benchmarks/
  ```

* **Graph RAG**:

  ```bash
  python grag.py --data_dir data/ --benchmarks benchmarks/
  ```

### Evaluation

After generating predictions, evaluate using:

```bash
python metrics_basemodel.py --predictions RAG_basemodel.csv --references benchmarks/...
python metrics.py --predictions RAG_evaluation_results.csv --references benchmarks/...
```

Metrics include Exact Match, ROUGE-1/2/L, BLEU, and cosine similarity.

### Visualizations

Generate radar plots to compare models:

```bash
python radar_plot.py --input_csv RAG_evaluation_results_with_normalized_scores.csv
```

Plots will be saved as PNG files (`model_comparison_radar_plot.png`, `model_evaluation_radar_plot.png`).

## Benchmarks

* **ALCE**: African Language Complex Evaluation
* **CRAG**: Complex Reasoning QA Benchmark
* **EXPERT2**: Expert-level QA set
* **Retrieval-QA-Benchmark**: Standard RAG QA evaluation
* **ScienceQA**: Science question answering dataset

Each benchmark provides a set of questions and reference answers for evaluation.

## Results

Performance summary plots are available in the repository root:

* `model_comparison_radar_plot.png`: Compares baseline vs. graph RAG across metrics.
* `model_evaluation_radar_plot.png`: Detailed model performance per benchmark.

## Contributing

Contributions, issues, and feature requests are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

