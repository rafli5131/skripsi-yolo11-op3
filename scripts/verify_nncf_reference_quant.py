"""Verify the no-nvcc NNCF QAT fake-quant backend on CPU and CUDA."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import torch  # noqa: E402

from nncf_reference_quant import PureTorchQuantizedFunctions, configure_nncf_reference_quantization  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this verification is intended for the QAT CUDA host.")

    report = configure_nncf_reference_quantization(force=True)

    # Exercise NNCF's actual custom autograd entry points, not only the backend
    # helper, so this catches an incorrect namespace patch before full training.
    from nncf.torch.quantization.quantize_functions import asymmetric_quantize, symmetric_quantize

    device = torch.device("cuda:0")
    x = torch.tensor(
        [[[-1.4, -0.5, 0.0, 0.65, 1.3]]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    scale = torch.tensor([1.25], dtype=torch.float32, device=device, requires_grad=True)
    sym = symmetric_quantize(x, 256, -128, 127, scale, 1e-6)
    sym.sum().backward(retain_graph=True)

    input_low = torch.tensor([-1.0], dtype=torch.float32, device=device, requires_grad=True)
    input_range = torch.tensor([2.0], dtype=torch.float32, device=device, requires_grad=True)
    asym = asymmetric_quantize(x, 256, 0, 255, input_low, input_range, 1e-6)
    asym.sum().backward()

    torch.cuda.synchronize(device)
    tensors = [sym, asym, x.grad, scale.grad, input_low.grad, input_range.grad]
    if any(value is None for value in tensors):
        raise RuntimeError("NNCF fake-quant autograd verification produced a missing tensor/gradient.")
    if not all(torch.isfinite(value).all() for value in tensors if value is not None):
        raise RuntimeError("NNCF fake-quant autograd verification produced non-finite values.")

    direct = PureTorchQuantizedFunctions.Quantize_forward(
        x.detach(), input_low.detach(), input_range.detach(), 256
    )
    if not torch.isfinite(direct).all():
        raise RuntimeError("Direct pure-PyTorch fake-quant backend produced non-finite values.")

    output = {
        "status": "passed",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "backend": report,
        "symmetric_forward_backward": True,
        "asymmetric_forward_backward": True,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
