"""
Setup script for Blackwell (RTX 5090) custom CUDA kernels
Compiles with CUDA 13.0 and native sm_120 support
"""

import os
import shutil

import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

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

is_windows = os.name == "nt"

torch_include_paths = torch.utils.cpp_extension.include_paths()

extra_cxx_args = [
    '/O2',
    '/std:c++17',
] if is_windows else [
    '-O3',
    '-std=c++17',
    '-fPIC',
]

common_nvcc_args = [
    '-O3',
    '-gencode=arch=compute_120,code=sm_120',
    '-gencode=arch=compute_120,code=compute_120',
    '--use_fast_math',
    '-std=c++17',
    '--expt-relaxed-constexpr',
    '--expt-extended-lambda',
    '--ptxas-options=-v',
    '--ptxas-options=-warn-spills',
    '--ptxas-options=-warn-lmem-usage',
    '-lineinfo',
    '--threads', '4',
    '-DCUDA_VERSION=13000',
    '-DBLACKWELL_ARCH=1',
]

if is_windows:
    common_nvcc_args.extend([
        '-Xcompiler', '/MD',
    ])
else:
    common_nvcc_args.extend([
        '-Xcompiler', '-fPIC',
        '-Xcompiler', '-Wall',
    ])

library_dir = os.path.join(cuda_home, 'lib', 'x64') if is_windows else os.path.join(cuda_home, 'lib64')
ninja_available = shutil.which('ninja') is not None

setup(
    name='blackwell_flux_kernels',
    version='1.0.0',
    description='Optimized CUDA kernels for RTX 5090 (Blackwell) FLUX training',
    author='FLUX Training Team',
    ext_modules=[
        CUDAExtension(
            name='blackwell_flux_kernels',
            sources=['blackwell_kernels.cu'],
            extra_compile_args={'cxx': extra_cxx_args, 'nvcc': common_nvcc_args},
            include_dirs=torch_include_paths + [os.path.join(cuda_home, 'include')],
            library_dirs=[library_dir],
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
            use_ninja=ninja_available,
            use_cuda=True,
        )
    },
    python_requires='>=3.11',
    install_requires=[
        'torch>=2.0.0',
    ],
)
