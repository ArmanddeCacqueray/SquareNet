import numpy as np


def from_backend(x):
    """Convert any supported array/tensor to a NumPy ndarray."""
    if isinstance(x, np.ndarray):
        return x
    module = type(x).__module__.split(".")[0]
    if module == "torch":
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_backend_device(x):
    """Return (backend, device) for a supported array."""
    if isinstance(x, np.ndarray):
        return "numpy", "cpu"
    module = type(x).__module__.split(".")[0]
    if module == "torch":
        return "torch", str(x.device)
    if module in ("jax", "jaxlib"):
        dev = getattr(x, "device", None)
        return "jax", str(dev) if dev is not None else "unknown"
    return "unknown", None


def to_backend(x, backend="numpy"):
    """
    Convert arrays between NumPy, Torch and JAX.

    Philosophy
    ----------
    - If backend already match, return x unchanged.
    - Otherwise always convert through NumPy.
    """
    current_backend, current_device = get_backend_device(x)
    # ------------------------------------------------------------------
    # Already on requested backend/device
    # ------------------------------------------------------------------
    if backend == current_backend:
        return x
    # ------------------------------------------------------------------
    # Convert through NumPy
    # ------------------------------------------------------------------
    arr = from_backend(x).copy()
    # ------------------------------------------------------------------
    # NumPy
    # ------------------------------------------------------------------
    if backend == "numpy":
        return arr
    # ------------------------------------------------------------------
    # Torch
    # ------------------------------------------------------------------
    if backend == "torch":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.as_tensor(x, device=torch.device(device))
    # ------------------------------------------------------------------
    # JAX
    # ------------------------------------------------------------------
    if backend == "jax":
        import jax.numpy as jnp
        out = jnp.asarray(arr)
        return out
    raise ValueError(
        f"Unknown backend '{backend}'. Expected 'numpy', 'torch' or 'jax'."
    )