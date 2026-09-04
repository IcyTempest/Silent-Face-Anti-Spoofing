# -*- coding: utf-8 -*-
"""Export a MiniFASNet .pth checkpoint to ONNX.

The architecture and input size are inferred from the checkpoint filename
(e.g. `2.7_80x80_MiniFASNetV2.pth`), reusing the repo's own parsing so the
exported graph matches exactly what `AntiSpoofPredict` runs at inference.

Usage:
    python export_onnx.py resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth
    python export_onnx.py <path/to.pth> --out model.onnx --opset 13 --check
"""
import os
import argparse
from collections import OrderedDict

import torch

from src.anti_spoof_predict import MODEL_MAPPING
from src.utility import get_kernel, parse_model_name


def load_model(model_path, device):
    model_name = os.path.basename(model_path)
    h_input, w_input, model_type, _ = parse_model_name(model_name)
    kernel_size = get_kernel(h_input, w_input)
    model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(device)

    state_dict = torch.load(model_path, map_location=device)
    # Strip a leading `module.` if the checkpoint was saved under DataParallel.
    first_key = next(iter(state_dict))
    if first_key.startswith("module."):
        state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
    model.load_state_dict(state_dict)
    model.eval()
    return model, h_input, w_input

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", help="path to the .pth checkpoint")
    ap.add_argument("--out", default=None, help="output .onnx path (default: alongside the .pth)")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--dynamic-batch", action="store_true",
                    help="allow a variable batch dimension in the exported graph")
    ap.add_argument("--check", action="store_true",
                    help="verify the ONNX output matches PyTorch (needs onnx + onnxruntime)")
    args = ap.parse_args()

    device = torch.device("cpu")  # export on CPU for a portable graph
    model, h, w = load_model(args.model_path, device)

    out_path = args.out or os.path.splitext(args.model_path)[0] + ".onnx"
    # NCHW, single channel-first RGB image, values already in [0,1] at inference.
    dummy = torch.randn(1, 3, h, w, device=device)

    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}} if args.dynamic_batch else None

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )
    # torch 2.x's exporter may split weights into an `<out>.data` sidecar (external
    # data). Consolidate into ONE self-contained file so deployment can't ship the
    # graph without its weights.
    import onnx as _onnx
    _m = _onnx.load(out_path)                       # pulls the sidecar back in
    _onnx.save(_m, out_path, save_as_external_data=False)
    _sidecar = out_path + ".data"
    if os.path.exists(_sidecar):
        os.remove(_sidecar)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"exported {args.model_path} -> {out_path}  "
          f"(input 1x3x{h}x{w}, opset {args.opset}, {size_mb:.2f} MB, single file)")

    if args.check:
        import numpy as np
        import onnx
        import onnxruntime as ort

        onnx.checker.check_model(onnx.load(out_path))
        with torch.no_grad():
            torch_out = model(dummy).cpu().numpy()
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
        max_diff = float(np.abs(torch_out - onnx_out).max())
        print(f"check: max abs diff PyTorch vs ONNX = {max_diff:.3e}",
              "OK" if max_diff < 1e-4 else "MISMATCH")
