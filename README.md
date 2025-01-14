# FP8 GEMM with PyTorch Interface

## Usage

Insall the kernels using the following commands:

```bash
git clone https://github.com/IST-DASLab/gemm_fp8.git
cd gemm_fp8
pip install -e .  # or pip install .
```

Then, the kernel can be used as follows:

```python
import torch
import gemm_fp8
y = gemm_fp8.matmul(a, b, alpha=1.0)
```

where `a` and `b` are the input matrices (in `torch.float8_e4m3fn` format) and `alpha` is the scaling factor (in `float`).

## Benchmark

Run the following command to benchmark the kernel:

```bash
python benchmark.py
```