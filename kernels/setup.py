"""
Setup script for Blackwell (RTX 5090) custom CUDA kernels
Compiles with CUDA 13.0 and native sm_120 support
"""

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch
import os

# Ensure CUDA 13.0 is used
cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
os.environ['CUDA_HOME'] = cuda_home

print("=" * 60)
print("BUILDING BLACKWELL CUSTOM KERNELS")
print("=" * 60)
print(f"CUDA Home: {cuda_home}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA version: {torch.version.cuda}")
print("=" * 60)

setup(
    name='blackwell_flux_kernels',
    version='1.0.0',
    description='Optimized CUDA kernels for RTX 5090 (Blackwell) FLUX training',
    author='FLUX Training Team',
    ext_modules=[
        CUDAExtension(
            name='blackwell_flux_kernels',
            sources=['blackwell_kernels.cu'],
            extra_compile_args={
                'cxx': [
                    '-O3',
                    '-std=c++17',
                    '-fPIC',
                ],
                'nvcc': [
                    '-O3',
                    # Native sm_120 for RTX 5090
                    '-gencode=arch=compute_120,code=sm_120',
                    # Also include compute_120 for JIT
                    '-gencode=arch=compute_120,code=compute_120',
                    # Fast math optimizations
                    '--use_fast_math',
                    # C++17 support
                    '-std=c++17',
                    # Extended lambdas and constexpr
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    # Verbose output
                    '--ptxas-options=-v',
                    # Warnings
                    '--ptxas-options=-warn-spills',
                    '--ptxas-options=-warn-lmem-usage',
                    # Debug info for profiling
                    '-lineinfo',
                    # Parallel compilation
                    '--threads', '4',
                    # Additional optimizations
                    '-Xcompiler', '-fPIC',
                    '-Xcompiler', '-Wall',
                    # Define CUDA version
                    '-DCUDA_VERSION=13000',
                    # Enable Blackwell features
                    '-DBLACKWELL_ARCH=1',
                ]
            },
            include_dirs=[
                torch.utils.cpp_extension.include_paths()[0],
                os.path.join(cuda_home, 'include'),
            ],
            library_dirs=[
                os.path.join(cuda_home, 'lib64'),
            ],
            libraries=[
                'cudart',
                'cublas',
                'cudnn',
                'cublasLt',
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(
            use_ninja=True,
            use_cuda=True
        )
    },
    python_requires='>=3.11',
    install_requires=[
        'torch>=2.0.0',
    ],
)
