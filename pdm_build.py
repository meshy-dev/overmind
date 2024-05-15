from torch.utils.cpp_extension import CppExtension, BuildExtension

def pdm_build_update_setup_kwargs(context, setup_kwargs):
    setup_kwargs.update(
        cmdclass={"build_ext": BuildExtension},
        ext_modules=[
            CppExtension("overmind._C", [
                "src/overmind/csrc/omhelpers.cpp"
            ])
        ],
    )
