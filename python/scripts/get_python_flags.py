#!/usr/bin/env python3

import os
import sys
import sysconfig
from pathlib import Path


def python_libdir_and_name():
    """
    Returns (libdir, libname_for_rustc)

    libdir: where libpythonX.Y.dylib lives
    libname_for_rustc: 'python3.X' (no 'lib' prefix, no extension)
    """
    libdir_list = sysconfig.get_config_vars("LIBDIR")
    soname_list = sysconfig.get_config_vars("INSTSONAME")

    if not libdir_list or not soname_list:
        raise RuntimeError("Couldn't get Python LIBDIR / INSTSONAME from sysconfig")

    libdir = libdir_list[0]  # e.g. /Users/me/.venv/lib
    soname = soname_list[0]  # e.g. libpython3.13.dylib

    name, ext = os.path.splitext(soname)  # name='libpython3.13', ext='.dylib'

    if name.startswith("lib"):
        name = name[3:]  # -> 'python3.13'

    return libdir, name


def torch_libdir():
    """
    Returns the directory that contains libtorch*.dylib, libc10.dylib, etc.
    This uses the currently active python (venv aware).
    """
    try:
        import torch  # type: ignore
    except Exception as e:
        # If torch isn't available, we just don't emit torch-related flags.
        # You might want to hard-fail instead depending on your project guarantees.
        return None

    torch_root = Path(torch.__file__).parent  # .../site-packages/torch
    libdir = torch_root / "lib"  # .../site-packages/torch/lib
    return str(libdir)


def emit_rustc_flags(py_libdir, py_libname, torch_lib):

    # 1. Link search path for Python
    print(f"cargo:rustc-link-search=native={py_libdir}")

    # 2. Link against libpython
    print(f"cargo:rustc-link-lib={py_libname}")

    # 3. Add rpath for Python's libdir so the final binary can find libpython3.X.dylib at runtime
    print(f"cargo:rustc-link-arg=-Wl,-rpath,{py_libdir}")

    # 4. Torch integration
    if torch_lib is not None:
        # Make rustc aware of torch's native libs at link time
        print(f"cargo:rustc-link-search=native={torch_lib}")

        # Add rpath for torch's lib dir so DYLD_LIBRARY_PATH isn't needed at runtime
        # Also, explicitly link to `libtorch_python`. This library is linked by `libtorch`, but because
        # the dependency is inirect, `rpath` will not propagate into `libtorch` and `libtorch_python`
        # will not be found at runtime (so we would still need to set up DYLD_LIBRARY_PATH without this step).
        print(f"cargo:rustc-link-arg=-Wl,-rpath,{torch_lib}")

        # Let downstream crates know we're using the PyTorch build of libtorch
        print("cargo:rustc-env=LIBTORCH_USE_PYTORCH=1")
    else:
        print("cargo:warning=Could not import torch in build environment", file=sys.stderr)
        sys.exit(1)

    # Tell Cargo when to rerun the build script:
    # - if the virtualenv changes
    # - or if this script changes (Cargo auto-handles build.rs, but you're shelling to Python)
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        print(f"cargo:rerun-if-env-changed=VIRTUAL_ENV={venv}")
    else:
        # Still emit something stable so Cargo knows about rerun-if-env-changed
        print("cargo:rerun-if-env-changed=VIRTUAL_ENV")

    # Always rerun if this script changes
    print(f"cargo:rerun-if-changed={__file__}")


if __name__ == "__main__":
    py_libdir, py_libname = python_libdir_and_name()
    torch_lib = torch_libdir()

    emit_rustc_flags(py_libdir, py_libname, torch_lib)
    # emit_python_config(CONFIG_FILE, py_libdir, py_libname, torch_lib)
