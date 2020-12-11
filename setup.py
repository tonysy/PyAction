#!/usr/bin/env python3
# Copyright (c) ShanghaiTech PLUS Group. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import glob
import os
from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 3], "Requires PyTorch >= 1.3"


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "pyaction", "layers", "csrc")

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        os.path.join(extensions_dir, "*.cu")
    )

    sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

        # It's better if pytorch can do this by default ..
        CC = os.environ.get("CC", None)
        if CC is not None:
            extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "pyaction._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


cur_dir = os.getcwd()
with open("tools/pyaction_run", "w") as pyaction_run:
    head = f"#!/bin/bash\n\nexport OMP_NUM_THREADS=1\n"
    pyaction_run.write(
        head + f"python3 {os.path.join(cur_dir, 'tools', 'run_net.py')} $@"
    )
with open("tools/pyaction_meta_run", "w") as pyaction_meta_run:
    head = f"#!/bin/bash\n\nexport OMP_NUM_THREADS=1\n"
    pyaction_meta_run.write(
        head + f"python3 {os.path.join(cur_dir, 'tools', 'meta_run_net.py')} $@"
    )
# with open("tools/pyaction_run_queue", "w") as pyaction_run:
#     head = f"""#!/bin/bash\n\nnum_gpu_require=4\nmax_gpu_num=4\necho "Wait until $num_gpu_require gpu are available..."\ngpu_info=($({os.path.join(cur_dir, 'tools', 'gpu_usage_scan.sh')} $num_gpu_require $max_gpu_num))\ngpu_num=${{gpu_info[0]}}\necho "got it, start the script!"\n\nexport OMP_NUM_THREADS=1\n"""  # noqa: E501

#     pyaction_run.write(
#         head + f"python3 {os.path.join(cur_dir, 'tools', 'run_net.py')} $@"
#     )

setup(
    name="pyaction",
    version="1.0",
    author="ShanghaiTech PLUS Group",
    url="unknown",
    description="A Toolkit for Video Understanding with PyTorch",
    python_requires=">=3.7",
    install_requires=[
        "av",
        "matplotlib",
        "termcolor>=1.1",
        "simplejson",
        "matplotlib",
        "colorama",
        "easydict",
        "pre-commit",
    ],
    packages=find_packages(exclude=("configs", "tests")),
    extras_require={"all": ["shapely", "psutil"]},
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    scripts=["tools/pyaction_run", "tools/pyaction_meta_run"],
)
