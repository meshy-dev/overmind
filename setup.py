from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    cmdclass={"build_ext": BuildExtension},
    ext_modules=[
        CppExtension("overmind._C", [
            "src/overmind/csrc/omhelpers.cpp"
        ])
    ],
)

