import ov_bootstrap_l  # required: bootstrap OpenVINO/NCS2 runtime

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import openvino as ov


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_shape(raw_shape, batch_size: int):
    """Resolve dynamic model input shape to concrete integers."""
    resolved = []
    for i, dim in enumerate(raw_shape):
        if i == 0:
            resolved.append(batch_size)
            continue

        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        else:
            fallback = [batch_size, 3, 256, 128]
            resolved.append(fallback[i] if i < len(fallback) else 1)

    return resolved


def make_dummy_input(input_shape, dtype):
    if np.issubdtype(dtype, np.integer):
        return np.random.randint(0, 255, size=input_shape, dtype=dtype)
    return np.random.randn(*input_shape).astype(dtype)


# ---------------------------------------------------------------------------
# Single Pipeline (Synchronous)
# ---------------------------------------------------------------------------

def benchmark_single_pipeline(compiled, input_any_name, dummy_input,
                              warmup, iterations, batch_size):
    """Classic synchronous inference — one request at a time."""
    infer_request = compiled.create_infer_request()

    print("\n[MODE] Single Pipeline (Synchronous)")
    print(f"[INFO] Warmup={warmup}, Iterations={iterations}")

    # Warm-up
    for _ in range(warmup):
        infer_request.infer({input_any_name: dummy_input})

    latencies_ms = []
    t_total_start = time.perf_counter()

    for _ in range(iterations):
        t0 = time.perf_counter()
        infer_request.infer({input_any_name: dummy_input})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    t_total_end = time.perf_counter()
    return latencies_ms, t_total_start, t_total_end


# ---------------------------------------------------------------------------
# Multi Pipeline (Asynchronous, NUM_REQUESTS parallel requests)
# ---------------------------------------------------------------------------

def benchmark_multi_pipeline(compiled, input_any_name, dummy_input,
                             warmup, iterations, batch_size, num_requests=3):
    """Async inference with *num_requests* in-flight requests for higher throughput."""
    infer_queue = [compiled.create_infer_request() for _ in range(num_requests)]

    print(f"\n[MODE] Multi Pipeline (Async × {num_requests} requests)")
    print(f"[INFO] Warmup={warmup}, Iterations={iterations}")

    # Warm-up — push requests round-robin
    for w in range(warmup):
        req = infer_queue[w % num_requests]
        req.start_async({input_any_name: dummy_input})
        req.wait()

    latencies_ms = []
    request_start_times = [0.0] * num_requests
    t_total_start = time.perf_counter()

    # Kick off initial in-flight requests
    initial_in_flight = min(num_requests, iterations)
    submitted = 0

    for i in range(initial_in_flight):
        request_start_times[i] = time.perf_counter()
        infer_queue[i].start_async({input_any_name: dummy_input})
        submitted += 1

    completed = 0
    while completed < iterations:
        req_idx = completed % num_requests
        req = infer_queue[req_idx]
        req.wait()

        t_done = time.perf_counter()
        latencies_ms.append((t_done - request_start_times[req_idx]) * 1000.0)
        completed += 1

        if submitted < iterations:
            request_start_times[req_idx] = time.perf_counter()
            req.start_async({input_any_name: dummy_input})
            submitted += 1

    t_total_end = time.perf_counter()
    return latencies_ms, t_total_start, t_total_end


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(title, latencies_ms, t_start, t_end, iterations, batch_size):
    total_sec = t_end - t_start
    avg_ms = statistics.mean(latencies_ms)
    p50_ms = float(np.percentile(latencies_ms, 50))
    p95_ms = float(np.percentile(latencies_ms, 95))
    p99_ms = float(np.percentile(latencies_ms, 99))
    throughput_fps = (iterations * batch_size) / total_sec if total_sec > 0 else 0.0

    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)
    print(f"  Total time       : {total_sec:.3f} s")
    print(f"  Average latency  : {avg_ms:.3f} ms")
    print(f"  P50 latency      : {p50_ms:.3f} ms")
    print(f"  P95 latency      : {p95_ms:.3f} ms")
    print(f"  P99 latency      : {p99_ms:.3f} ms")
    print(f"  Throughput       : {throughput_fps:.2f} samples / s")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------

def benchmark_openvino_ncs2_reid(model_path: Path, device: str, warmup: int,
                                  iterations: int, batch_size: int,
                                  pipeline: str, num_requests: int):
    core = ov.Core()

    available_devices = core.available_devices
    print(f"[INFO] OpenVINO available devices: {available_devices}")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading ReID model: {model_path}")

    model = core.read_model(str(model_path))
    compiled = core.compile_model(model, device)

    input_port = compiled.input(0)
    input_any_name = input_port.get_any_name()
    input_shape = resolve_shape(input_port.shape, batch_size=batch_size)

    ov_dtype = input_port.get_element_type()
    np_dtype = ov_dtype.to_dtype()

    print(f"[INFO] Input name : {input_any_name}")
    print(f"[INFO] Input shape: {input_shape}")
    print(f"[INFO] Input dtype: {np_dtype}")

    dummy_input = make_dummy_input(input_shape, np_dtype)

    if pipeline == "single":
        lats, t0, t1 = benchmark_single_pipeline(
            compiled, input_any_name, dummy_input,
            warmup, iterations, batch_size,
        )
        print_report("NCS2 ReID Benchmark — Single Pipeline (Sync)",
                      lats, t0, t1, iterations, batch_size)

    elif pipeline == "multi":
        lats, t0, t1 = benchmark_multi_pipeline(
            compiled, input_any_name, dummy_input,
            warmup, iterations, batch_size,
            num_requests=num_requests,
        )
        print_report(f"NCS2 ReID Benchmark — Multi Pipeline (Async ×{num_requests})",
                      lats, t0, t1, iterations, batch_size)

    elif pipeline == "both":
        # Run single first
        lats_s, t0_s, t1_s = benchmark_single_pipeline(
            compiled, input_any_name, dummy_input,
            warmup, iterations, batch_size,
        )
        print_report("NCS2 ReID Benchmark — Single Pipeline (Sync)",
                      lats_s, t0_s, t1_s, iterations, batch_size)

        # Then multi
        lats_m, t0_m, t1_m = benchmark_multi_pipeline(
            compiled, input_any_name, dummy_input,
            warmup, iterations, batch_size,
            num_requests=num_requests,
        )
        print_report(f"NCS2 ReID Benchmark — Multi Pipeline (Async ×{num_requests})",
                      lats_m, t0_m, t1_m, iterations, batch_size)

        # Comparison
        avg_s = statistics.mean(lats_s)
        avg_m = statistics.mean(lats_m)
        tp_s = (iterations * batch_size) / (t1_s - t0_s) if (t1_s - t0_s) > 0 else 0
        tp_m = (iterations * batch_size) / (t1_m - t0_m) if (t1_m - t0_m) > 0 else 0

        print("\n" + "-" * 50)
        print("  Comparison: Single vs Multi Pipeline")
        print("-" * 50)
        print(f"  Avg latency  : {avg_s:.3f} ms  vs  {avg_m:.3f} ms")
        print(f"  Throughput   : {tp_s:.2f} samples / s  vs  {tp_m:.2f} samples / s")
        if tp_s > 0:
            print(f"  Speedup      : {tp_m / tp_s:.2f}×")
        print("-" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark OSNet ReID (OpenVINO IR) on NCS2 — single & multi pipeline"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("osnet_ReID_onnx/osnet_ir_0_25_ncs2/osnet_x0_25_msmt17_ncs2_batch1.xml"),
        help="Path to OpenVINO IR xml model (default: osnet_x0_25 NCS2 batch1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="MYRIAD",
        help='OpenVINO device, e.g. "MYRIAD"',
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["single", "multi", "both"],
        default="both",
        help='Pipeline mode: "single", "multi", or "both" (default: both)',
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=3,
        help="Number of async infer requests for multi pipeline (default: 3)",
    )
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=200, help="Benchmark iterations")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    benchmark_openvino_ncs2_reid(
        model_path=args.model,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iters,
        batch_size=args.batch,
        pipeline=args.pipeline,
        num_requests=args.num_requests,
    )


if __name__ == "__main__":
    main()
