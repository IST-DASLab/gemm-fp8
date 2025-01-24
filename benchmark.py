import torch
import gemm_fp8
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Iterable, List, Tuple

sns.set()

iters = 100
warmup = 5


def to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)



def make_rand_tensors(dtype: torch.dtype, m: int, n: int,
                      k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    a = torch.randn((m, k), device='cuda').contiguous() * 5
    b = torch.randn((n, k), device='cuda').contiguous() * 5

    if dtype == torch.float8_e4m3fn:
        return to_fp8(a), to_fp8(b)
    if dtype == torch.bfloat16:
        return a.to(torch.bfloat16), b.to(torch.bfloat16)
    if dtype == torch.float16:
        return a.half(), b.half()
    if dtype == torch.float32:
        return a.float(), b.float()

    raise ValueError("unsupported dtype")


# bench
def bench_fn(fn: Callable, *args, **kwargs) -> Tuple:

    times_ = []
    for i in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    for _ in range(10):
        start = time.time()
        for i in range(iters):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        times_.append((time.time() - start) * 1000 / iters)

    return np.mean(np.array(times_)), np.std(np.array(times_))




K_lists = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 11008]
last_dim = 4096

token_dim = [512, 1024, 2048, 4096, 8192, 16384]
dim_lists = [ #from LLaMa3-8B
             [4096,  4096],
             [4096,  14336],
             [14336, 4096]
             ]


plt.figure(figsize=(15, 10))

for token in token_dim:
    x_labels = []
    fp8_speedups = []

    for n_, k_ in dim_lists:
        m_ = token

        x_labels.append(f"{k_}x{n_}")

        a, b = make_rand_tensors(torch.bfloat16, m_, n_, k_)
        a_fp8, b_fp8 = to_fp8(a), to_fp8(b)

        print("---- m: ", m_, "k: ", k_, "n: ", n_, "----")

        bf16_times, bf16_times_std = bench_fn(torch.matmul, a, b.t())
        cutlass_times, cutlass_times_std = bench_fn(gemm_fp8.matmul, a_fp8, b_fp8, 1.0)#, False)

        fp8_speedups.append(bf16_times/cutlass_times)
        print(f"Speedup (FP8): {(bf16_times/cutlass_times):.2f}x")

    plt.plot(x_labels, fp8_speedups, 'o--', label=f"M={token} FP8")

plt.plot(x_labels, np.ones(len(x_labels))*2, "k")
plt.axhline(1, color='black', linestyle='--')
plt.xlabel("KxN")
plt.ylabel("Speedup")
plt.title(f"Speedup of FP8 over BF16 ({torch.cuda.get_device_name(0)})")
plt.legend()
plt.savefig("benchmark_fp8.png")




