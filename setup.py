import os
import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension
from utils import extension

source_file = []
source_file += glob.glob(os.path.join("./ads/common/ops/csrc/", "*.cpp"))
source_file += glob.glob(os.path.join("./bind/", "*.cpp"))

exts = []
ext1 = extension.NpuExtension(
    name="ads_c",
    sources=source_file,
)

exts.append(ext1)
setup(
    name="ads",
    version="1.0",
    description='Cpp Extension Include ascend_ads',
    keywords='ads',
    ext_modules=exts,
    author='Ascend Contributors',
    cmdclass={"build_ext": BuildExtension},
    packages=find_packages()
)
