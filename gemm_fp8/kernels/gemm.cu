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

//#include "helper.h"
//#include "hopper_fp8_commandline.hpp"

#include <gemm.h>

using namespace cute;

using         ElementA    = cutlass::float_e4m3_t;
using         LayoutA     = cutlass::layout::RowMajor;
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;

using         ElementB    = cutlass::float_e4m3_t;
using         LayoutB     = cutlass::layout::ColumnMajor;
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;

// FIXME: necessary?
using         ElementC    = cutlass::bfloat16_t;
using         LayoutC     = cutlass::layout::RowMajor;
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;

// D matrix configuration
using         ElementD    = ElementC;
using         LayoutD     = LayoutC;
constexpr int AlignmentD  = AlignmentC;

// Auxiliary matrix configuration and other fusion types
using         ElementAux   = ElementC;
using         LayoutAux    = LayoutC;
using         ElementAmax  = float;
using         ElementBias  = float;

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementCompute      = float;                                          // Element type for epilogue computation
using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
using TileShape           = Shape<_128,_128,_128>;                          // Threadblock-level tile size
using ClusterShape        = Shape<_1,_2,_1>;                                // Shape of the threadblocks in a cluster
using KernelSchedule      = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSchedule    = cutlass::epilogue::TmaWarpSpecializedCooperative;
using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;
/* using FusionOperation     = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltActAmaxAux<
                                    LayoutAux,
                                    cutlass::epilogue::thread::ReLU,
                                    ElementD,
                                    ElementCompute,
                                    ElementAux,
                                    ElementAmax,
                                    ElementBias,
                                    ElementC>; */

/* using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
                                    ArchTag, OperatorClass,
                                    TileShape, ClusterShape,
                                    EpilogueTileType,
                                    ElementAccumulator, ElementCompute,
                                    ElementC, LayoutC, AlignmentC,
                                    ElementD, LayoutD, AlignmentD,
                                    EpilogueSchedule,
                                    FusionOperation
                                  >::CollectiveOp; */

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
                                    ArchTag, OperatorClass,
                                    TileShape, ClusterShape,
                                    EpilogueTileType,
                                    ElementAccumulator, ElementCompute,
                                    ElementC, LayoutC, AlignmentC,
                                    ElementD, LayoutD, AlignmentD,
                                    cutlass::epilogue::collective::EpilogueScheduleAuto
                                  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
                                    ArchTag, OperatorClass,
                                    ElementA, LayoutA, AlignmentA,
                                    ElementB, LayoutB, AlignmentB,
                                    ElementAccumulator,
                                    TileShape, ClusterShape,
                                    cutlass::gemm::collective::StageCountAutoCarveout<
                                      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
                                    >,
                                    //KernelSchedule
                                    cutlass::gemm::collective::KernelScheduleAuto
                                  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
                                    Shape<int,int,int,int>, // Indicates ProblemShape
                                    CollectiveMainloop,
                                    CollectiveEpilogue
                                  >;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Extract information from Gemm kernel.
using EpilogueOutputOp  = typename Gemm::EpilogueOutputOp;
using ElementScalar     = typename EpilogueOutputOp::ElementScalar;
//using ElementAmax       = typename EpilogueOutputOp::ElementAmax;
//using ActivationFunctor = typename EpilogueOutputOp::ActivationFn;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using StrideAux = StrideD;

constexpr bool IsDFp8 =
    cute::is_same_v<ElementD, cutlass::float_e4m3_t> or
    cute::is_same_v<ElementD, cutlass::float_e5m2_t>;

constexpr bool IsAuxFp8 =
    cute::is_same_v<ElementAux, cutlass::float_e4m3_t> or
    cute::is_same_v<ElementAux, cutlass::float_e5m2_t>;

StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
StrideAux stride_aux;

using LayoutScalar = cutlass::layout::PackedVectorLayout;
cutlass::HostTensor<ElementScalar, LayoutScalar> scalar_alpha;
cutlass::HostTensor<ElementScalar, LayoutScalar> scalar_beta;
/* cutlass::HostTensor<ElementScalar, LayoutScalar> scale_A;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_B;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_C;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_D;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_aux;
cutlass::HostTensor<ElementAmax  , LayoutScalar> abs_max_D;
cutlass::HostTensor<ElementAmax  , LayoutScalar> reference_abs_max_D;
cutlass::HostTensor<ElementAmax  , LayoutScalar> abs_max_aux;
cutlass::HostTensor<ElementAmax  , LayoutScalar> reference_abs_max_aux; */

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

// Command line options parsing
template<typename RasterOrderOptions>
struct Options {

  cutlass::gemm::GemmCoord problem_size;

  float alpha = 1.f, beta = 0.f;
  float scale_a = 1.f, scale_b = 1.f, scale_c = 1.f, scale_d = 1.f, scale_aux = 1.f;
  bool device_scale = false;
  bool save_aux = false;
  bool save_amax = false;
  RasterOrderOptions raster = RasterOrderOptions::AlongN;
  int swizzle = 1;

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

  //using ElementCompute = typename Gemm::GemmKernel::Epilogue::OutputOp::ElementCompute;

  uint64_t seed;


  //
  // Methods
  //

  TestbedRunner() { }


  bool run(
    const Options<RasterOrderOptions>& options,
    torch::Tensor out,  // FP32/FP16/BF16 (TODO)
    torch::Tensor x,    // float_e4m3_t
    torch::Tensor y     // float_e4m3_t
    )
  {

    /* scalar_alpha.resize(cutlass::make_Coord(1));
    scalar_beta.resize(cutlass::make_Coord(1));
    cutlass::reference::host::TensorFill(scalar_alpha.host_view(), options.alpha);
    cutlass::reference::host::TensorFill(scalar_beta.host_view(), options.beta);
    scalar_alpha.sync_device();
    scalar_beta.sync_device(); */

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.problem_size.m(), options.problem_size.k(), 1));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(options.problem_size.k(), options.problem_size.n(), 1));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(options.problem_size.m(), options.problem_size.n(), 1));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.problem_size.m(), options.problem_size.n(), 1));
    stride_aux = stride_D;

    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.problem_size.m(), options.problem_size.n(), options.problem_size.k(), 1},
      {static_cast<ElementA*>(x.data_ptr()), stride_A,
       static_cast<ElementB*>(y.data_ptr()), stride_B},
      {
        {}, // epilogue.thread
        static_cast<ElementC*>(out.data_ptr()), stride_C,
        static_cast<ElementD*>(out.data_ptr()), stride_D
      }
    };

    auto &fusion_args = arguments.epilogue.thread;
    fusion_args.alpha = options.alpha;
    fusion_args.beta = options.beta;
    fusion_args.alpha_ptr = scalar_alpha.device_data();
    fusion_args.beta_ptr = scalar_beta.device_data();

    /* fusion_args.scale_a = 1.f;
    fusion_args.scale_b = 1.f;
    fusion_args.scale_c = 1.f;
    fusion_args.scale_a_ptr = scale_A.device_data();
    fusion_args.scale_b_ptr = scale_B.device_data();
    fusion_args.scale_c_ptr = scale_C.device_data(); */

    // ignored if tensor types are not fp8
    /* fusion_args.scale_d = 1.f;
    fusion_args.scale_aux = 1.f;
    fusion_args.scale_d_ptr   = scale_D.device_data();
    fusion_args.scale_aux_ptr = scale_aux.device_data(); */

    // leaving/setting these as nullptr disables the fusion at runtime
    //fusion_args.bias_ptr = nullptr;

    arguments.scheduler.raster_order = options.raster;
    // The tile scheduler will swizzle up to 8 and with the nearest multiple of 2 (i.e., 1, 2, 4, and 8)
    arguments.scheduler.max_swizzle_size = options.swizzle;

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

  Options<RasterOrderOptions> options(M, N, K, alpha);

  TestbedRunner<Gemm> testbed_fast_accum;
  return testbed_fast_accum.run(options, out, x, y);
}