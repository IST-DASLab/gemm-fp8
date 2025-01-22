from setuptools import setup
import torch.utils.cpp_extension as torch_cpp_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import pathlib, torch
setup_dir = os.path.dirname(os.path.realpath(__file__))
HERE = pathlib.Path(__file__).absolute().parent
torch_version = torch.__version__

def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__',
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass

def get_cuda_arch_flags():
    return [
        '-gencode', 'arch=compute_90a,code=sm_90a',  # Hopper
        '--expt-relaxed-constexpr'
    ]

def third_party_cmake():
    import subprocess, sys, shutil

    cmake = shutil.which('cmake')
    if cmake is None:
            raise RuntimeError('Cannot find CMake executable.')

    retcode = subprocess.call([cmake, HERE])
    if retcode != 0:
        sys.stderr.write("Error: CMake configuration failed.\n")
        sys.exit(1)

if __name__ == '__main__':

    #TODO: add CUDA version >= 12 check

    assert torch.cuda.is_available(), "CUDA is not available!"
    device = torch.cuda.current_device()
    print(f"Current device: {torch.cuda.get_device_name(device)}")
    print(f"Current CUDA capability: {torch.cuda.get_device_capability(device)}")
    #assert torch.cuda.get_device_capability(device)[0] >= 9, f"CUDA capability must be >= 9.0, yours is {torch.cuda.get_device_capability(device)}"

    # Check if version is higher than 2.0
    print(f"PyTorch version: {torch_version}")
    assert int(torch_version.split('.')[0]) >= 2, "Torch version should be higher than 2!"


    third_party_cmake()
    remove_unwanted_pytorch_nvcc_flags()
    setup(
        name='gemm_fp8',
        ext_modules=[
            CUDAExtension(
                name='gemm_fp8._CUDA',
                sources=[
                    'gemm_fp8/kernels/bindings.cpp',
                    'gemm_fp8/kernels/gemm.cu',
                ],
                include_dirs=[
                    os.path.join(setup_dir, 'gemm_fp8/kernels/include'),
                    os.path.join(setup_dir, 'cutlass/include'),
                    os.path.join(setup_dir, 'cutlass/tools/util/include'),
                    "/mnt/nfs/clustersw/shared/cuda/12.4.1/include"
                ],
                extra_compile_args={
                    'cxx': [],
                    'nvcc': get_cuda_arch_flags(),
                },
                libraries=["cuda", "cudart"],  # Ensure these are linked
                library_dirs=["/mnt/nfs/clustersw/shared/cuda/12.4.1/lib64"],
                extra_link_args=[
                '-L/mnt/nfs/clustersw/shared/cuda/12.4.1/lib64',
                '-lcudart',
                '-lcuda',
                ]
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
