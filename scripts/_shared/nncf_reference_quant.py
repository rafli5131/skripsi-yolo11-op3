"""Pure-PyTorch fake-quant backend for NNCF 2.13 QAT on CUDA hosts without nvcc.

NNCF 2.13 normally lazy-compiles custom CPU/CUDA quantization extensions.  On
managed GPU servers it is common to have a working PyTorch CUDA runtime and GPU
driver but no CUDA development toolkit (``nvcc``) or system Python development
headers.  In that situation NNCF's CUDA extension loader raises
``nncf.InstallationError`` before QAT can start.

This module replaces only the *execution backend* of NNCF fake quantization
with a pure-PyTorch implementation of the same reference equations used by
NNCF 2.13.  It does not call ``nncf.quantize`` and does not turn QAT into PTQ.
The NNCF graph, quantizers, trainable ranges, compression controller, checkpoint
format, and QAT training lifecycle remain unchanged.

The private ExtensionNamespace hook is intentionally version-guarded to
NNCF 2.13.x.  A deterministic numerical self-check against NNCF's NumPy
reference implementation is executed before the fallback is activated.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from typing import Any

import nncf
import numpy as np
import torch
from nncf.torch.quantization.reference import ReferenceBackendType, ReferenceQuantize
from nncf.torch.utils import sum_like


@dataclass(frozen=True)
class BackendReport:
    backend: str
    nncf_version: str
    nvcc: str | None
    cpu_namespace_patched: bool
    cuda_namespace_patched: bool
    numerical_self_check: bool
    cuda_smoke_check: bool


class PureTorchQuantizedFunctions:
    """Drop-in namespace implementing NNCF Quantize_forward/backward in torch."""

    @staticmethod
    def Quantize_forward(
        input_: torch.Tensor,
        input_low: torch.Tensor,
        input_range: torch.Tensor,
        levels: int,
    ) -> torch.Tensor:
        scale = (levels - 1) / input_range
        input_high = input_low + input_range
        clipped = torch.minimum(torch.maximum(input_, input_low), input_high)
        zero_point = torch.round(-input_low * scale)
        output = (clipped - input_low) * scale
        output = output - zero_point
        output = torch.round(output)
        return output / scale

    @classmethod
    def Quantize_backward(
        cls,
        grad_output: torch.Tensor,
        input_: torch.Tensor,
        input_low: torch.Tensor,
        input_range: torch.Tensor,
        levels: int,
        level_low: int,
        level_high: int,
        is_asymmetric: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ``is_asymmetric`` exists for compatibility with NNCF's CPU extension
        # signature.  NNCF's own reference implementation does not use it.
        del is_asymmetric
        output = cls.Quantize_forward(input_, input_low, input_range, levels)

        mask_hi = (input_ > (input_low + input_range)).to(input_.dtype)
        mask_lo = (input_ < input_low).to(input_.dtype)
        mask_in = 1 - mask_hi - mask_lo

        range_sign = torch.sign(input_range)
        err = (output - input_) * torch.reciprocal(input_range * range_sign)
        grad_range = grad_output * (
            err * mask_in + range_sign * (level_low / level_high) * mask_lo + mask_hi
        )
        grad_range = sum_like(grad_range, input_range)

        grad_input = grad_output * mask_in
        grad_low = sum_like(grad_output * (mask_hi + mask_lo), input_low)
        return grad_input, grad_low, grad_range


def _assert_supported_nncf() -> None:
    version = str(nncf.__version__)
    if not version.startswith("2.13."):
        raise RuntimeError(
            "The no-nvcc QAT reference backend is intentionally pinned to NNCF 2.13.x; "
            f"found NNCF {version}. Review NNCF internals before using this fallback on another version."
        )


def _numerical_self_check() -> None:
    """Compare forward/backward with NNCF's NumPy reference equations."""
    reference = ReferenceQuantize(backend_type=ReferenceBackendType.NUMPY)

    input_np = np.array(
        [
            [
                [[-2.0, -0.75, 0.10], [0.30, 0.95, 1.80]],
                [[-1.25, -0.40, 0.00], [0.35, 0.80, 1.40]],
            ]
        ],
        dtype=np.float32,
    )
    low_np = np.array([[[[-1.0]], [[-0.75]]]], dtype=np.float32)
    range_np = np.array([[[[2.0]], [[1.5]]]], dtype=np.float32)
    grad_np = np.array(
        [
            [
                [[0.25, 0.50, 0.75], [1.00, 1.25, 1.50]],
                [[1.50, 1.25, 1.00], [0.75, 0.50, 0.25]],
            ]
        ],
        dtype=np.float32,
    )

    input_t = torch.from_numpy(input_np.copy())
    low_t = torch.from_numpy(low_np.copy())
    range_t = torch.from_numpy(range_np.copy())
    grad_t = torch.from_numpy(grad_np.copy())

    for level_low, level_high in ((0, 255), (-128, 127)):
        levels = 256
        ref_out = reference.forward(input_np.copy(), low_np.copy(), range_np.copy(), levels)
        ref_grads = reference.backward(
            grad_np.copy(),
            input_np.copy(),
            low_np.copy(),
            range_np.copy(),
            ref_out.copy(),
            level_low,
            level_high,
        )

        torch_out = PureTorchQuantizedFunctions.Quantize_forward(input_t, low_t, range_t, levels)
        torch_grads = PureTorchQuantizedFunctions.Quantize_backward(
            grad_t, input_t, low_t, range_t, levels, level_low, level_high
        )

        np.testing.assert_allclose(torch_out.numpy(), ref_out, rtol=1e-5, atol=1e-6)
        for actual, expected in zip(torch_grads, ref_grads):
            np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-6)


def _cuda_smoke_check() -> bool:
    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda:0")
    x = torch.tensor([[-1.25, -0.1, 0.4, 1.6]], dtype=torch.float32, device=device)
    low = torch.tensor([-1.0], dtype=torch.float32, device=device)
    input_range = torch.tensor([2.0], dtype=torch.float32, device=device)
    grad = torch.ones_like(x)
    out = PureTorchQuantizedFunctions.Quantize_forward(x, low, input_range, 256)
    grads = PureTorchQuantizedFunctions.Quantize_backward(grad, x, low, input_range, 256, -128, 127)
    torch.cuda.synchronize(device)
    if not torch.isfinite(out).all() or not all(torch.isfinite(item).all() for item in grads):
        raise RuntimeError("Pure-PyTorch NNCF reference backend produced non-finite CUDA values.")
    return True


def configure_nncf_reference_quantization(*, force: bool = False) -> dict[str, Any]:
    """Activate the pure-PyTorch NNCF fake-quant backend when needed.

    ``force=True`` is used by the managed-server QAT launcher so every tuning
    candidate runs with exactly the same fake-quant execution backend.  With
    ``force=False`` the fallback activates only when ``nvcc`` is unavailable or
    ``NNCF_QAT_REFERENCE_BACKEND=1`` is set.
    """
    _assert_supported_nncf()

    nvcc = shutil.which("nvcc")
    env_force = os.environ.get("NNCF_QAT_REFERENCE_BACKEND", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "reference",
    }
    use_reference = force or env_force or nvcc is None
    if not use_reference:
        report = BackendReport(
            backend="native-nncf-extension",
            nncf_version=str(nncf.__version__),
            nvcc=nvcc,
            cpu_namespace_patched=False,
            cuda_namespace_patched=False,
            numerical_self_check=False,
            cuda_smoke_check=False,
        )
        return asdict(report)

    _numerical_self_check()

    # Mutate the already-imported lazy ExtensionNamespace objects in place.
    # quantize_functions.py imports references to these same objects, so no
    # module reload or source-package modification is required.
    from nncf.torch.quantization import extensions as nncf_extensions

    cpu_namespace = getattr(nncf_extensions, "QuantizedFunctionsCPU", None)
    cuda_namespace = getattr(nncf_extensions, "QuantizedFunctionsCUDA", None)

    cpu_patched = hasattr(cpu_namespace, "_loaded_namespace")
    if cpu_patched:
        cpu_namespace._loaded_namespace = PureTorchQuantizedFunctions

    cuda_patched = hasattr(cuda_namespace, "_loaded_namespace")
    if torch.cuda.is_available() and not cuda_patched:
        raise RuntimeError(
            "NNCF CUDA quantization namespace is not patchable on this NNCF build; "
            "refusing to continue with an unverified workaround."
        )
    if cuda_patched:
        cuda_namespace._loaded_namespace = PureTorchQuantizedFunctions

    cuda_ok = _cuda_smoke_check()
    report = BackendReport(
        backend="pure-pytorch-nncf-reference",
        nncf_version=str(nncf.__version__),
        nvcc=nvcc,
        cpu_namespace_patched=cpu_patched,
        cuda_namespace_patched=cuda_patched,
        numerical_self_check=True,
        cuda_smoke_check=cuda_ok,
    )
    return asdict(report)
