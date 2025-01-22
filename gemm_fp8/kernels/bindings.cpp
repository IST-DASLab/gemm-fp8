#include <torch/extension.h>
#include <gemm.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <utility> // For std::pair

torch::Tensor fp8_matmul(
    const torch::Tensor &X,  const torch::Tensor &Y, const float alpha
)
{
    torch::checkAllContiguous("fp8_matmul", {{X, "X",       0},
                                                {Y, "Y", 1}});
    torch::checkDeviceType("fp8_matmul", {X, Y}, at::DeviceType::CUDA);

    torch::checkAllSameGPU("fp8_matmul", {{X, "X",       0},
                                          {   Y, "Y", 1}});
    uint32_t M = X.size(0);
    uint32_t N = Y.size(0);
    auto OUT = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(X.device()));

    fp8_matmul_host(OUT, X, Y, alpha);

    return OUT;
}

//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
)
{
    m.def("fp8_matmul", &fp8_matmul,
        "fp8_matmul");
}