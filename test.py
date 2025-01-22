import torch
from typing import Callable, Iterable, List, Tuple
import gemm_fp8


def to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)

def to_int8(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def make_rand_tensors(dtype: torch.dtype, m: int, n: int,
                      k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    a = torch.randn((m, k), device='cuda') * 5
    b = torch.randn((n, k), device='cuda').t() * 5

    if dtype == torch.int8:
        return to_int8(a).contiguous(), to_int8(b).contiguous()
    if dtype == torch.float8_e4m3fn:
        return to_fp8(a).contiguous(), to_fp8(b).contiguous()

    raise ValueError("unsupported dtype")


m = 1024
n = 1024
k = 1024

x_fp8, y_fp8 = make_rand_tensors(torch.float8_e4m3fn, m, n, k)

print('------------------- cutlass -------------------')
out = gemm_fp8.matmul(x_fp8, y_fp8, 2.0)
print(out[0])
print(out[0][0:16])
print(out[0].shape)

print('------------------- baseline -------------------')
baseline_out = torch.matmul(x_fp8.half(), y_fp8.half().t()).to(torch.bfloat16)
print(2.0*baseline_out[0])
print(2.0*baseline_out[0][0:16])
print(baseline_out[0].shape)



