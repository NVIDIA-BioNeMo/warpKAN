# warpKAN


A high-performance GPU-accelerated library for Kolmogorov-Arnold Networks (KANs) built with NVIDIA Warp, featuring efficient B-spline evaluation and support for Unbounded KANs (UKANs).

## Key Features

GPU Acceleration: Built with NVIDIA Warp for optimal GPU performance

Efficient B-spline Evaluation: Reduces computational complexity from O(K·dg·din·dout) to O(K·din·dout)

Memory Optimization: Handles grid sizes up to 2^18 (1000x larger than naive implementations)

Performance Gains: 5.5-15x faster than PyTorch implementations

Unbounded Domains: Support for UKANs that operate without fixed grid boundaries

PyTorch Integration: Seamless PyTorch bindings for easy adoption

## Install

```bash
pip install warpkan
pip install -e .
```


## Cite
If you use this repository, please cite our paper:
```bash
@misc{moradzadeh2025wanttrainkansscale,
      title={Want to train KANS at scale? Now UKAN!}, 
      author={Alireza Moradzadeh and Srimukh Prasad Veccham and Lukasz Wawrzyniak and Miles Macklin and Saee G. Paliwal},
      year={2025},
      eprint={2408.11200},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.11200}, 
}
```
