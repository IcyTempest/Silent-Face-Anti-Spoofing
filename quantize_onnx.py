# -*- coding: utf-8 -*-
"""Static INT8 quantization for an exported MiniFASNet ONNX model.

Calibration feeds the model REAL face crops preprocessed byte-for-byte the way
`AntiSpoofPredict.predict` does (detect -> CropImage.crop at the model's scale ->
ToTensor). That is what makes the INT8 ranges match inference; calibrating on raw
frames instead is the usual reason quantized anti-spoof models lose accuracy.

    python quantize_onnx.py resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.onnx \
        --calib-dir real_world_data_processed --num-calib 200 --compare

Requires: onnxruntime (with onnxruntime.quantization). The matching .pth must sit
next to the .onnx so the filename -> (scale, size) parsing lines up.
"""
import os
import glob
import argparse

import cv2
import numpy as np

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.data_io import transform as trans
from src.utility import parse_model_name

from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
from onnxruntime.quantization.shape_inference import quant_pre_process


def preprocess(frame, bbox, cropper, scale, h, w, to_tensor):
    """One BGR frame -> (1,3,h,w) float32, identical to AntiSpoofPredict.predict."""
    param = {"org_img": frame, "bbox": bbox, "scale": scale,
             "out_w": w, "out_h": h, "crop": scale is not None}
    img = cropper.crop(**param)
    return to_tensor(img).unsqueeze(0).numpy().astype(np.float32)


def gather_inputs(calib_dir, num, predictor, cropper, scale, h, w, to_tensor):
    paths = sorted(glob.glob(os.path.join(calib_dir, "**", "*.*"), recursive=True))
    paths = [p for p in paths if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not paths:
        raise SystemExit(f"no images found under {calib_dir}")
    rng = np.random.default_rng(0)
    rng.shuffle(paths)

    inputs = []
    for p in paths:
        if len(inputs) >= num:
            break
        frame = cv2.imread(p)
        if frame is None:
            continue
        bbox = predictor.get_bbox(frame)
        inputs.append(preprocess(frame, bbox, cropper, scale, h, w, to_tensor))
    if not inputs:
        raise SystemExit("could not build any calibration inputs")
    print(f"calibration: {len(inputs)} crops from {calib_dir}")
    return inputs


class Reader(CalibrationDataReader):
    def __init__(self, inputs, input_name):
        self._it = iter([{input_name: x} for x in inputs])

    def get_next(self):
        return next(self._it, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_path", help="fp32 .onnx to quantize (a matching .pth must sit beside it)")
    ap.add_argument("--out", default=None, help="output path (default: <name>.fp16.onnx)")
    ap.add_argument("--calib-dir", default="real_world_data_processed")
    ap.add_argument("--num-calib", type=int, default=200)
    ap.add_argument("--compare", action="store_true",
                    help="report fp32-vs-int8 output agreement on the calibration crops")
    args = ap.parse_args()

    name = os.path.basename(args.onnx_path)
    h, w, _, scale = parse_model_name(name)          # e.g. 2.7_80x80_MiniFASNetV2 -> scale 2.7, 80x80
    out_path = args.out or args.onnx_path.replace(".onnx", ".fp16.onnx")

    predictor = AntiSpoofPredict(0)                  # only used for get_bbox (YuNet)
    cropper = CropImage()
    to_tensor = trans.Compose([trans.ToTensor()])

    inputs = gather_inputs(args.calib_dir, args.num_calib, predictor, cropper,
                           scale, h, w, to_tensor)

    # ORT wants shape inference + optimization before static quant.
    prep = args.onnx_path.replace(".onnx", ".prep.onnx")
    quant_pre_process(args.onnx_path, prep)

    import onnxruntime as ort
    input_name = ort.InferenceSession(prep, providers=["CPUExecutionProvider"]).get_inputs()[0].name

    quantize_static(
        prep, out_path,
        calibration_data_reader=Reader(inputs, input_name),
        quant_format=QuantFormat.QDQ,      # portable; ORT mobile / most backends prefer QDQ
        per_channel=True,                  # per-channel conv weights -> much less accuracy loss
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    os.remove(prep)

    fp32_mb = os.path.getsize(args.onnx_path) / 1e6
    fp16_mb = os.path.getsize(out_path) / 1e6
    print(f"quantized -> {out_path}")
    print(f"size: {fp32_mb:.3f} MB -> {fp16_mb:.3f} MB  ({fp32_mb / fp16_mb:.2f}x smaller)")

    if args.compare:
        s32 = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])
        s8 = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        n32 = s32.get_inputs()[0].name
        n8 = s8.get_inputs()[0].name
        agree, maxdiff = 0, 0.0
        for x in inputs:
            a = s32.run(None, {n32: x})[0]
            b = s8.run(None, {n8: x})[0]
            agree += int(np.argmax(a) == np.argmax(b))
            maxdiff = max(maxdiff, float(np.abs(_softmax(a) - _softmax(b)).max()))
        n = len(inputs)
        print(f"argmax agreement fp32 vs int8: {agree}/{n} ({100 * agree / n:.1f}%)")
        print(f"max softmax-prob difference:   {maxdiff:.4f}")


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


if __name__ == "__main__":
    main()
