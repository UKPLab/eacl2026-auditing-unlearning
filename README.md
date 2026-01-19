# Auditing Language Model Unlearning via Information Decomposition

Code for the EACL 2026 paper: **Auditing Language Model Unlearning via Information Decomposition**.

We introduce a framework to audit machine unlearning by decomposing the information in model representations into **Unlearned Knowledge** (unique to base model) and **Residual Knowledge** (shared/redundant).

## Features
- **RINE Implementation**: Estimates Redundant Information using Lagrangian optimization over neural probes.
- **PID Metrics**: Calculates Unlearned ($I^B_{uniq}$) and Residual ($I_{\cap}$) knowledge.
- **Risk Score**: Implements the inference-time abstention mechanism described in Section 6.

## File structure
```
eacl2026-auditing-unlearning/
├── README.md
├── requirements.txt
├── src/
│   ├── rine.py           # Core logic: RINE implementation (Section 5.1 & App A.3)
│   └── probe_all_layers.py        # Linear probing script (Section 3)
├── run_audit.py   # Main entry point to run the audit
```

## Setup and Installation
We use the [open-unlearning](https://github.com/locuslab/open-unlearning) framework to train and use the base and unlearned models. 

```bash
conda create -n audit_unlearning python=3.10
conda activate audit_unlearning
pip install -r requirements.txt
```

## Usage
### Probing representations
Results from Section 3 can be reproduced by running `python src/probe_all_layers.py`. 

This script will extarct activations from the base and unlearned model, train logistic probes on them and save the AUROC performance of the probes at each layer.

### Run the audit
To audit a model (e.g., Llama-3-8b unlearned via RMU) against its base version:
```python
python audit_unlearning.py \
    --base_model "<path to your base model>" \
    --unlearned_model "<path to your unlearned model>" \
    --dataset_name "locuslab/TOFU" \
    --forget_subset "forget10" \
    --retain_subset "retain90" \
    --exp_name "<experiment_identifier>" \
    --cache_dir "./activations"
```

The script can be used to audit unlearned models as well as compute the risk scores from our paper.

## Cite

Please use the following citation:

```
@inproceedings{goel2025auditing,
  title={Auditing Language Model Unlearning via Information Decomposition},
  author={Goel, Anmol and Ritter, Alan and Gurevych, Iryna},
  booktitle={Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  year={2026}
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
