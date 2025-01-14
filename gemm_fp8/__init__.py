import torch
import gemm_fp8._CUDA


__all__ = [ 
           "matmul"
           ]

def matmul(x: torch.Tensor, 
           y: torch.Tensor, 
           alpha: float = 1.0, 
           fastAcc: bool = True) -> torch.Tensor:
    '''
    Matrix-Matrix Multiplication for FP8 data type in the form of (x @ y.t())*alpha.
    The output is BF16 data type. todo: support arbitrary output dtype!
    Argumengs:
        x: torch.Tensor, shape (M, K)
        y: torch.Tensor, shape (K, N)
        alpha: float, which is multiplied by the output (default=1.0)
        fastAcc: bool, (default=True)
    '''
    if fastAcc:
        return gemm_fp8._CUDA.fp8_matmul_fastAcc(x, y, alpha)
    else:
        return gemm_fp8._CUDA.fp8_matmul(x, y, alpha)
    