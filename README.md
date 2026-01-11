# Auditing Language Model Unlearning via Information Decomposition

Code for the EACL 2025 paper: **Auditing Language Model Unlearning via Information Decomposition**.

We introduce a framework to audit machine unlearning by decomposing the information in model representations into **Unlearned Knowledge** (unique to base model) and **Residual Knowledge** (shared/redundant).

## Features
- **RINE Implementation**: Estimates Redundant Information using Lagrangian optimization over neural probes.
- **PID Metrics**: Calculates Unlearned ($I^B_{uniq}$) and Residual ($I_{\cap}$) knowledge.
- **Risk Score**: Implements the inference-time abstention mechanism described in Section 6.

## Installation

```bash
pip install -r requirements.txt
```


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
