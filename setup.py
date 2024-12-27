from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext



ext_modules = [
    Pybind11Extension(
        "efficient_attention",
        ["EfficientAttention.cpp"],  # El nombre de tu archivo C++
    ),
]

setup(
    name="efficient_attention",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

