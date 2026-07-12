import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time

import numpy as np
import torch

from processor.detector.p2pnet_trt import P2PNetTRTDetector


def bench_trt(engine, raw_img_size, input_img_size, device, iters, warmup):
    detector = P2PNetTRTDetector(
        engine_path=engine,
        device=device,
        threshold=0.5,
        raw_img_size=raw_img_size,
        trt_img_size=input_img_size,
    )

    frame = np.random.randint(
        0, 255, (input_img_size[0], input_img_size[1], 3), dtype=np.uint8
    )

    t = []
    for i in range(warmup + iters):
        torch.cuda.synchronize()
        s = time.perf_counter()
        result = detector.infer(frame)
        torch.cuda.synchronize()
        e = time.perf_counter()

        if i >= warmup:
            t.append((e - s) * 1000)

    def stat(name, xs):
        xs = np.array(xs)
        print(
            f"  {name:12s} mean={xs.mean():7.2f} ms  min={xs.min():7.2f}  max={xs.max():7.2f}"
        )

    stat("inference [TRT]", t)
    print(f"  approx FPS = {1000.0 / np.array(t).mean():.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine", type=str, default="weights/p2pnet/cutout_4k_fused.engine"
    )
    parser.add_argument("--raw_img_size", type=int, nargs=2, default=[4320, 7680])
    parser.add_argument("--input_img_size", type=int, nargs=2, default=[2160, 3840])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    bench_trt(
        args.engine,
        args.raw_img_size,
        args.input_img_size,
        args.device,
        args.iters,
        args.warmup,
    )


if __name__ == "__main__":
    main()
