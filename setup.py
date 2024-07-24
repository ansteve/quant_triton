from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_marlin_cuda',
    ext_modules=[
        CUDAExtension(
            'my_marlin_cuda',
            [
                'extension/marlin/marlin_cuda.cpp',
                'extension/marlin/marlin_cuda_kernel.cu'
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
