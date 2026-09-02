import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def resolve_input_shape(raw_shape, batch_size: int):
    """Resolve dynamic/unknown dimensions to concrete integers for benchmarking."""
    shape = list(raw_shape)
    resolved = []
    for i, dim in enumerate(shape):
        if i == 0:
            resolved.append(batch_size)
            continue
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        else:
            fallback = [batch_size, 3, 256, 128]
            resolved.append(fallback[i] if i < len(fallback) else 1)
    return resolved


def benchmark_cpu_onnx(
    model_path: Path,
    warmup: int,
    iterations: int,
    batch_size: int,
    intra_threads: int,
    inter_threads: int,
):
    print(f"[INFO] model={model_path}")
    print("[INFO] provider=CPUExecutionProvider")

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = intra_threads
    sess_options.inter_op_num_threads = inter_threads
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = resolve_input_shape(input_meta.shape, batch_size=batch_size)

    print(f"[INFO] input_name={input_name}")
    print(f"[INFO] input_shape={input_shape}")
    print(
        f"[INFO] intra_threads={intra_threads}, inter_threads={inter_threads}, "
        f"warmup={warmup}, iterations={iterations}"
    )

    dummy = np.random.randn(*input_shape).astype(np.float32)

    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy})

    latencies_ms = []
    start_total = time.perf_counter()

    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    end_total = time.perf_counter()

    total_sec = end_total - start_total
    avg_ms = statistics.mean(latencies_ms)
    p50_ms = np.percentile(latencies_ms, 50)
    p95_ms = np.percentile(latencies_ms, 95)
    p99_ms = np.percentile(latencies_ms, 99)
    throughput = (iterations * batch_size) / total_sec if total_sec > 0 else 0.0

    print("\n" + "=" * 42)
    print("CPU ONNX Benchmark Result")
    print("=" * 42)
    print(f"Average latency : {avg_ms:.3f} ms")
    print(f"P50 latency     : {p50_ms:.3f} ms")
    print(f"P95 latency     : {p95_ms:.3f} ms")
    print(f"P99 latency     : {p99_ms:.3f} ms")
    print(f"Throughput      : {throughput:.2f} FPS")
    print("=" * 42)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark ONNX ReID model on CPU")
    parser.add_argument("--model", type=Path, required=True, help="Path to ONNX model")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--intra-threads", type=int, default=2, help="ONNXRuntime intra-op threads")
    parser.add_argument("--inter-threads", type=int, default=1, help="ONNXRuntime inter-op threads")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    benchmark_cpu_onnx(
        model_path=args.model,
        warmup=args.warmup,
        iterations=args.iters,
        batch_size=args.batch,
        intra_threads=args.intra_threads,
        inter_threads=args.inter_threads,
    )


if __name__ == "__main__":
    main()
    # Example usage:
    # python benchmark_onnx_on_cpu.py --model osnet_x0_25_msmt17_combineall_test.onnx --warmup 10 --iters 100 --batch 1 --intra-threads 2 --inter-threads 1
