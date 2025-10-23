"""
Python bindings to the `torch` create
"""


def load_zrm_torch():
    """
    The generated library needs to be named zrm_torch.so to load,
    but Rust generates `libzrm_torch.so` or `libzrm_torch.dylib`.
    We can create a symlink, but that is not very flexible.
    Instead, we try to load the library from the build directory
    unless it is not already present in the environment.
    """

    try:
        import zrm_torch

        return zrm_torch
    except ImportError:
        from ctypes import CDLL
        from os.path import dirname, join, isfile

        build_dir = join(dirname(__file__), "../../../")
        files = [
            "target/release/libzrm_torch.so",
            "target/release/libzrm_torch.dylib",
            "target/debug/libzrm_torch.so",
            "target/debug/libzrm_torch.dylib",
        ]
        for fl in files:
            path = join(build_dir, fl)
            if isfile(path):
                import importlib.machinery
                import importlib.util

                if path.endswith(".dylib"):
                    loader = importlib.machinery.ExtensionFileLoader("zrm_torch", path)
                    spec = importlib.util.spec_from_loader(loader.name, loader)
                else:
                    spec = importlib.util.spec_from_file_location("zrm_torch", path)
                    loader = spec.loader

                if spec is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                loader.exec_module(mod)

                # from sys import modules
                # modules["libzrm_torch"] = mod

                return mod

    # just do the import again so that a sensible
    # exception is raised
    import zrm_torch


# NOTE: this must go first
libzrm_torch = load_zrm_torch()

# export `Term` and `to_terms`
from .term import Term, to_terms, eq, le, lt, ge, gt, ifelse, var
