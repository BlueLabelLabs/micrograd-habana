import torch
from functools import wraps
import importlib.util


def get_device_initial(preferred_device=None):
    """
    Determine the appropriate device to use (cuda, hpu, or cpu).
    Args:
        preferred_device (str): User-preferred device ('cuda', 'hpu', or 'cpu').

    Returns:
        str: Device string ('cuda', 'hpu', or 'cpu').
    """
    # Check for HPU support
    if importlib.util.find_spec("habana_frameworks") is not None:
        from habana_frameworks.torch.utils.library_loader import load_habana_module

        load_habana_module()
        if torch.hpu.is_available():
            if preferred_device == "hpu" or preferred_device is None:
                return "hpu"
        else:
            raise RuntimeError("HPU is not available. Please check Habana setup.")

    # Check for CUDA (GPU support)
    if torch.cuda.is_available():
        if preferred_device == "cuda" or preferred_device is None:
            return "cuda"

    # Default to CPU
    return "cpu"


def auto_tensorize(cls):
    """
    Class decorator that wraps every non-special method so that
    each argument is converted to a tensor via `self.backend`.
    """
    for attr_name, attr_value in list(vars(cls).items()):
        # Only wrap callable attributes that aren't special (dunder) methods
        if callable(attr_value) and not attr_name.startswith("__"):

            def make_wrapped_method(method):
                @wraps(method)
                def wrapped_method(self, *args, **kwargs):
                    # Convert positional args to backend tensors
                    new_args = [self.backend.create_tensor(a) for a in args]

                    # Convert keyword args to backend tensors
                    new_kwargs = {
                        k: self.backend.create_tensor(v) for k, v in kwargs.items()
                    }

                    # Now call the original method with converted arguments
                    return method(self, *new_args, **new_kwargs)

                return wrapped_method

            wrapped = make_wrapped_method(attr_value)
            setattr(cls, attr_name, wrapped)

    return cls
