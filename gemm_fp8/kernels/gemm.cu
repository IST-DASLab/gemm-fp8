/*! \file
    \brief Example of running an Ada FP8 GEMM.

    In addition to using FP8 Tensor Core instructions, the Ada FP8 GEMM uses a distinct epilogue
    that enables additional scaling of operands/outputs, storing a pre-activation-function output
    tensor (called the "auxiliary" output), and computing the absolute maximum value of the
    outputs.

    Pseudocode for this epilogue is as follows:

    Aux = ((alpha * scale_a * scale_b) * accumulator) + ((beta * scale_c) * source) + bias
    D = activation(Aux)

    if Aux is fp8 type:
        abs_max_output = max( abs(aux) | (for every aux in Aux))
        Aux = scale_aux * Aux
    endif

    if D is fp8 type:
        abs_max_output = max( abs(d) | (for every d in D))
        D = scale_d * D
    endif

    Parameter Aux is optionally stored to global memory
*/

#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gemm.h"

#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic_with_scaling.h"
#include "cutlass/gemm/device/gemm_universal_with_absmax.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include <gemm.h>

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAuxOutput = ElementOutput;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
static int const kStages = 3;
static int const kAlignmentA = 16;
static int const kAlignmentB = 16;

using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity,
    ElementOutput,
    ElementAuxOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    //8,
    ElementAccumulator,
    ElementAccumulator
    >;

template <typename MathOperator, typename TileShape, typename WarpShape>
using Gemm_ = cutlass::gemm::device::GemmUniversalWithAbsMax<
    ElementA,
    LayoutA, // Row-major
    ElementB,
    LayoutB, // Column-major
    ElementOutput,
    LayoutC, // Row-major
    ElementAccumulator, // float
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    TileShape,
    WarpShape,
    cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages,
    kAlignmentA,
    kAlignmentB,
    MathOperator
  >;

using ElementAbsmax = typename EpilogueOutputOp::ElementAbsmax;


// Command line options parsing
struct Options {

  cutlass::gemm::GemmCoord problem_size;

  float alpha;
  float beta;

  Options(int M, int N, int K, float scale=1.f):
    beta(0.f)
  {
    problem_size = cutlass::gemm::GemmCoord{M, N, K};
    alpha = scale;
  }

  /// Compute performance in GFLOP/s
  float gflops(float runtime_s) const {
    // Two flops per multiply-add
    return 2.0f * float(problem_size.product()) / float(1.0e9) / runtime_s;
  }
};

/// Helper class to run the kernel
template <typename Gemm>
struct TestbedRunner {

  using ElementCompute = typename Gemm::GemmKernel::Epilogue::OutputOp::ElementCompute;

  uint64_t seed;


  //
  // Methods
  //

  TestbedRunner() { }


  bool run(
    Options& options,
    torch::Tensor out,  // FP32/FP16/BF16 (TODO)
    torch::Tensor x,    // float_e4m3_t
    torch::Tensor y     // float_e4m3_t
    )
  {


    //
    // Initialize the GEMM operator
    //

    typename Gemm::EpilogueOutputOp::Params::ActivationParams activation_params{
      ElementCompute(options.alpha),
      ElementCompute(options.beta)
    };

    typename Gemm::EpilogueOutputOp::Params epilogue_params{
      activation_params,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    };

    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      options.problem_size,
      /* batch_count = */ 1,
      epilogue_params,

      reinterpret_cast<ElementA*>(x.data_ptr()),
      reinterpret_cast<ElementB*>(y.data_ptr()),
      reinterpret_cast<ElementOutput*>(out.data_ptr()),
      reinterpret_cast<ElementOutput*>(out.data_ptr()),

      nullptr,
      nullptr,

      options.problem_size.m() * options.problem_size.k(),
      options.problem_size.n() * options.problem_size.k(),
      options.problem_size.m() * options.problem_size.n(),
      options.problem_size.m() * options.problem_size.n(),
      (int)options.problem_size.m(), // Batch stride vector

      x.stride(0),
      y.stride(0),
      out.stride(0),
      out.stride(0),
      (int64_t)0 // Leading dimension of vector. This must be 0
    };

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::can_implement() failed" << std::endl;
      return false;
    }

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::initialize() failed" << std::endl;
      return false;
    }

    //
    // Run the GEMM
    //

    status = gemm_op();

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::run() failed" << std::endl;
      return false;
    }

    return true;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

bool fp8_matmul_host(
    torch::Tensor out,  // FP32
    torch::Tensor x,    // float_e4m3_t
    torch::Tensor y,    // float_e4m3_t
    float alpha
){
  auto M = x.size(0);
  auto N = y.size(0);
  auto K = x.size(1);

  Options options(M, N, K, alpha);

  if (K==4096 && N==4096){
    using TileShape = typename cutlass::gemm::GemmShape<128, 64, 128>;
    using WarpShape = typename cutlass::gemm::GemmShape<64, 32, 128>;
    TestbedRunner<Gemm_<cutlass::arch::OpMultiplyAdd, TileShape, WarpShape>> testbed_fast_accum;
    return testbed_fast_accum.run(options, out, x, y);
  } else {
    using TileShape = typename cutlass::gemm::GemmShape<64, 128, 64>;
    using WarpShape = typename cutlass::gemm::GemmShape<32, 64, 64>;
    TestbedRunner<Gemm_<cutlass::arch::OpMultiplyAdd, TileShape, WarpShape>> testbed_fast_accum;
    return testbed_fast_accum.run(options, out, x, y);
  }
}


bool fp8_matmul_fastAcc_host(
    torch::Tensor out,  // FP32
    torch::Tensor x,    // float_e4m3_t
    torch::Tensor y,    // float_e4m3_t
    float alpha
){
  auto M = x.size(0);
  auto N = y.size(0);
  auto K = x.size(1);

  Options options(M, N, K, alpha);


  if (K==4096 && N==4096){
    using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
    TestbedRunner<Gemm_<cutlass::arch::OpMultiplyAddFastAccum, TileShape, WarpShape>> testbed_fast_accum;
    return testbed_fast_accum.run(options, out, x, y);
  } else {
    using TileShape = typename cutlass::gemm::GemmShape<64, 128, 64>;
    using WarpShape = typename cutlass::gemm::GemmShape<32, 64, 64>;
    TestbedRunner<Gemm_<cutlass::arch::OpMultiplyAddFastAccum, TileShape, WarpShape>> testbed_fast_accum;
    return testbed_fast_accum.run(options, out, x, y);
  }
}