# Transformer-GB [NeurIPS'24]

This repository contains the implementation of our paper "Learning to Compute Gröbner bases" [NeurIPS'24].

## Requirements

- SageMath 10.0+
- Python 3.8+
- PyTorch 2.0+
- transformers

## SageMath Setup

SageMath 10.0+ cannot be installed using `apt-get` currently (Nov, 12, 2024). 
Follow the instruction in [this page](https://sagemanifolds.obspm.fr/install_ubuntu.html). 

## Project Structure

```
Transformer-GB/
├── sh/             # Scripts for running experiments
│   ├── prepare_config.sh     # Configuration preparation
│   ├── build_all_datasets.sh # Dataset generation
│   ├── train_all.sh         # Model training
│   ├── eval_generation.sh   # Model evaluation
│   ├── timing_all.sh        # Performance benchmarking
│   ├── showcase_all.sh      # Example cases
│   └── collect_all_results.sh # Results aggregation
├── config/         # Dataset generation configuration
├── data/           # Generated datasets
├── results/        # Training and experimental results
├── src/            # Source code for the Transformer model
└── notebook/       # Jupyter notebooks with example usage
```

## Usage

To reproduce our experiments, follow these steps:

### 1. Dataset Generation
```bash
# Generate configuration files
./sh/prepare_config.sh

# Build all datasets
./sh/build_all_datasets.sh
```

### 2. Model Training
```bash
# Train Transformer models
./sh/train_all.sh
```

### 3. Evaluation and Analysis
```bash
# Evaluate model performance
./sh/eval_generation.sh

# Benchmark computation times
./sh/timing_all.sh

# Generate showcase examples
./sh/showcase_all.sh

# Collect and aggregate results
./sh/collect_all_results.sh
```

**Important Notes:**
- These scripts perform exhaustive experiments across all parameter combinations (number of variables, coefficient fields, etc.)
- For initial testing, we recommend starting with:
  - Number of variables: 2
  - Field: GF7
  - Encoding: standard

## License

This project is licensed under the Apache 2.0 License.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{
kera2024learning,
title={Learning to compute Gr\"obner bases},
author={Hiroshi Kera and Yuki Ishihara and Yuta Kambe and Tristan Vaccon and Kazuhiro Yokoyama},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=ZRz7XlxBzQ}
}
```
