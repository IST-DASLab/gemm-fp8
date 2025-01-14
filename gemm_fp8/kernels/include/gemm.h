#pragma once
#include <common.h>
#include <torch/types.h>

bool fp8_matmul_fastAcc_host(
    torch::Tensor out,  // BF16
    torch::Tensor x,  // float_e4m3_t
    torch::Tensor y, // float_e4m3_t
    float alpha
);

bool fp8_matmul_host(
    torch::Tensor out,  // BF16
    torch::Tensor x,  // float_e4m3_t
    torch::Tensor y, // float_e4m3_t
    float alpha
);