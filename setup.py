import os
import glob
from setuptools import setup, find_packages
# from torch_npu.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension
import cpp_extension

source_file = []
source_file += glob.glob(os.path.join("./common/ops/csrc/", "*.cpp"))
source_file += glob.glob(os.path.join("./common/ops/", "*.cpp"))

exts = []
ext1 = cpp_extension.NpuExtension(
    name="ads",
    sources=source_file,
)

exts.append(ext1)
setup(
    name="ads",
    version="1.0",
    description='Cpp Extension Include ascend_accelerator',
    keywords='ads',
    ext_modules=exts,
    author='Ascend Contributors',
    cmdclass={"build_ext": BuildExtension},
    packages=find_packages()
)
