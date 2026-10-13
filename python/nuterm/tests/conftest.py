# torch must load before the zrth C-extension (libtorch rpath); importing it
# here ensures the right order regardless of a test module's own import order.
import torch  # noqa: F401
