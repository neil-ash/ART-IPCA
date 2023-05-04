# ART-IPCA

## Overview

Code for ICLR '23 Tiny Paper "A Simple, Fast Algorithm for Continual Learning from High-Dimensional Data"

Paper available here: https://openreview.net/forum?id=TPTbHxeR6U

## Files

- [model.py](https://github.com/neil-ash/ART-IPCA/blob/main/model.py): model described in Algorithm $1$ of the paper implemented in sklearn-like interface
- [all_metrics.py](https://github.com/neil-ash/ART-IPCA/blob/main/all_metrics.py): code for sequential training and testing, outputs 'meta-table' of results on task $i$ after learning task $j$
- [demo.ipynb](https://github.com/neil-ash/ART-IPCA/blob/main/demo.ipynb): jupyter notebook demonstrating functionality on the MNIST dataset

## Citation
```
@misc{
ashtekar2023a,
title={A Simple, Fast Algorithm for Continual Learning from High-Dimensional Data},
author={Neil Ashtekar and Vasant G Honavar},
year={2023},
url={https://openreview.net/forum?id=TPTbHxeR6U}
}
```

## Questions?

Please reach out to nca5096@psu.edu
