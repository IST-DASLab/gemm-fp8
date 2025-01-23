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
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gett.hpp"

#include <gemm.h>

using namespace cute;

using         ElementA    = cutlass::float_e4m3_t;
using         LayoutA     = cutlass::layout::RowMajor;
constexpr int AlignmentA  = 16 / sizeof(ElementA);

using         ElementB    = cutlass::float_e4m3_t;
using         LayoutB     = cutlass::layout::ColumnMajor;
constexpr int AlignmentB  = 16 / sizeof(ElementB);

using         ElementC    = cutlass::bfloat16_t;
using         LayoutC     = cutlass::layout::RowMajor;
constexpr int AlignmentC  = 16 / sizeof(ElementC);

// Core kernel configurations
using ElementAccumulator  = float;
using ElementComputeEpilogue = float;
using ArchTag             = cutlass::arch::Sm90;
using OperatorClass       = cutlass::arch::OpClassTensorOp;
using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;

using DefaultSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
using PongSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using FastDefaultSchedule =
    cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
using FastPongSchedule =
    cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;

template <bool PONG>
using SlowAccum = cute::conditional_t<PONG, PongSchedule, DefaultSchedule>;

template <bool PONG>
using FastAccum =
      cute::conditional_t<PONG, FastPongSchedule, FastDefaultSchedule>;

template <bool PONG, bool FAST_ACCUM>
using MainLoopSchedule =
      cute::conditional_t<FAST_ACCUM, FastAccum<PONG>, SlowAccum<PONG>>;

using Scale_ =
      cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementComputeEpilogue>;
using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::multiplies,
    ElementC,
    ElementComputeEpilogue,
    cutlass::FloatRoundStyle::round_to_nearest>;
using EpilogueEVT =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, Scale_, Accum>;

template <typename TileShape, typename ClusterShape>
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
                                    ArchTag,
                                    OperatorClass,
                                    TileShape,
                                    ClusterShape,
                                    EpilogueTileType,
                                    ElementAccumulator,
                                    ElementComputeEpilogue,
                                    ElementC, LayoutC, AlignmentC,
                                    ElementC, LayoutC, AlignmentC,
                                    cutlass::epilogue::TmaWarpSpecialized,
                                    EpilogueEVT>::CollectiveOp;

template <typename TileShape, typename ClusterShape, bool PONG, bool FAST_ACCUM>
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
                                    ArchTag,
                                    OperatorClass,
                                    ElementA, LayoutA, AlignmentA,
                                    ElementB, LayoutB, AlignmentB,
                                    ElementAccumulator,
                                    TileShape,
                                    ClusterShape,
                                    cutlass::gemm::collective::StageCountAutoCarveout<
                                            static_cast<int>(
                                              sizeof(typename CollectiveEpilogue<TileShape, ClusterShape>::SharedStorage))>,
                                    MainLoopSchedule<PONG, FAST_ACCUM>>::CollectiveOp;

template <typename TileShape, typename ClusterShape, bool PONG, bool FAST_ACCUM>
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
                                    Shape<int,int,int>,
                                    CollectiveMainloop<TileShape, ClusterShape, PONG, FAST_ACCUM>,
                                    CollectiveEpilogue<TileShape, ClusterShape>
                                  >;

template <typename TileShape, typename ClusterShape, bool PONG, bool FAST_ACCUM>
using Gemm_ = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel<TileShape, ClusterShape, PONG, FAST_ACCUM>>;

// Command line options parsing
struct Options {

  cutlass::gemm::GemmCoord problem_size;

  float alpha = 1.f, beta = 0.f;

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

  uint64_t seed;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;

  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;

  //
  // Methods
  //

  TestbedRunner() { }


  bool run(
    const Options& options,
    torch::Tensor out,  // FP32/FP16/BF16 (TODO)
    torch::Tensor x,    // float_e4m3_t
    torch::Tensor y     // float_e4m3_t
    )
  {

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.problem_size.m(), options.problem_size.k(), 1));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(options.problem_size.n(), options.problem_size.k(), 1));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(options.problem_size.m(), options.problem_size.n(), 1));

    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.problem_size.m(), options.problem_size.n(), options.problem_size.k()},
      {reinterpret_cast<ElementA*>(x.data_ptr()), stride_A,
       reinterpret_cast<ElementB*>(y.data_ptr()), stride_B},
      {
        {}, // epilogue.thread
        (ElementC*)out.data_ptr<at::BFloat16>(), stride_C,
        (ElementC*)out.data_ptr<at::BFloat16>(), stride_C
      }
    };

    arguments.epilogue.thread = {
      {float(options.alpha)}, // scale
      {}, // Accumulator
      {}, // Multiplies
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

  /* if(M==512){
    using TileShape           = Shape<_128,_128,_128>;
    using ClusterShape        = Shape<_1,_2,_1>;
    TestbedRunner<Gemm_<TileShape, ClusterShape, false, true>> testbed_fast_accum;
    return testbed_fast_accum.run(options, out, x, y);
  } else {
    using TileShape           = Shape<_128,_128,_128>;
    using ClusterShape        = Shape<_2,_1,_1>;
    TestbedRunner<Gemm_<TileShape, ClusterShape, true, true>> testbed_fast_accum;
    return testbed_fast_accum.run(options, out, x, y);
  } */
  using TileShape           = Shape<_128,_128,_128>;
  using ClusterShape        = Shape<_2,_1,_1>;
  TestbedRunner<Gemm_<TileShape, ClusterShape, true, true>> testbed_fast_accum;
  return testbed_fast_accum.run(options, out, x, y);
}