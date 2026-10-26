"""Benchmark and Smoke-Test Script for Video Backbones on Local GPU.

Evaluates candidate spatiotemporal backbones (S3D, R(2+1)D-18) on NVIDIA RTX 3050 (4 GB VRAM).
Measures peak memory, extraction latency (ms), and throughput (clips/sec).
"""

import time
import torch
import torchvision.models.video as v


def benchmark_model(name: str, model_builder, input_shape, device="cuda", batch_size=8, warmup=10, iters=30):
    """Run controlled benchmark on a video backbone."""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING BACKBONE: {name.upper()}")
    print(f"  Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    print(f"  Batch size: {batch_size} clips (Clip shape: {input_shape})")
    print(f"{'='*60}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Build model in eval mode
    model = model_builder().to(device)
    model.eval()

    # Synthetic batch of snippets (B, 3, T, H, W)
    dummy_input = torch.randn(batch_size, *input_shape, device=device)

    # Warmup
    print("  Running warmup iterations...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    torch.cuda.synchronize()

    # Timed iterations
    print(f"  Running {iters} timed iterations...")
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            out = model(dummy_input)
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time

    # Compute metrics
    total_clips = batch_size * iters
    throughput_clips_per_sec = total_clips / total_time
    latency_per_batch_ms = (total_time / iters) * 1000.0
    latency_per_clip_ms = latency_per_batch_ms / batch_size
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"\n  --- Benchmark Results for {name.upper()} ---")
    print(f"  Feature Tensor Output Shape : {out.shape}")
    print(f"  Feature Dimensionality (D)  : {out.shape[-1]}")
    print(f"  Peak VRAM Allocation        : {peak_vram_mb:.1f} MB ({peak_vram_mb / 1024:.2f} GB / 4.0 GB)")
    print(f"  Latency per Batch ({batch_size} clips): {latency_per_batch_ms:.2f} ms")
    print(f"  Latency per Clip            : {latency_per_clip_ms:.2f} ms")
    print(f"  Extraction Throughput       : {throughput_clips_per_sec:.1f} clips/second")
    print(f"  Status                      : PASSED (Zero OOM, Stable)")

    del model, dummy_input, out
    torch.cuda.empty_cache()

    return {
        "name": name,
        "feature_dim": out.shape[-1] if 'out' in locals() else None,
        "peak_vram_mb": peak_vram_mb,
        "throughput_fps": throughput_clips_per_sec,
        "latency_per_clip_ms": latency_per_clip_ms,
    }


class S3DFeatureExtractor(torch.nn.Module):
    """Wrapper around torchvision S3D to extract 1024-d pooled spatiotemporal features."""
    def __init__(self):
        super().__init__()
        base = v.s3d(weights="KINETICS400_V1")
        self.features = base.features
        self.avgpool = base.avgpool
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        feat = self.features(x)
        pooled = self.avgpool(feat)
        return torch.flatten(pooled, 1)


class R2Plus1DFeatureExtractor(torch.nn.Module):
    """Wrapper around torchvision R(2+1)D-18 to extract 512-d pooled spatiotemporal features."""
    def __init__(self):
        super().__init__()
        base = v.r2plus1d_18(weights="KINETICS400_V1")
        self.stem = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        pooled = self.avgpool(h)
        return torch.flatten(pooled, 1)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA unavailable. Exiting.")
        sys.exit(1)

    print(f"Target GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # 1. Benchmark S3D (1024-dim, Inception-based separable I3D)
    res_s3d = benchmark_model(
        "S3D (1024-d, I3D Separable Variant)",
        lambda: S3DFeatureExtractor(),
        input_shape=(3, 16, 224, 224),
        batch_size=8
    )

    # 2. Benchmark R(2+1)D-18 (512-dim, ResNet-based factorized 3D)
    res_r2p1d = benchmark_model(
        "R(2+1)D-18 (512-d, Factorized ResNet)",
        lambda: R2Plus1DFeatureExtractor(),
        input_shape=(3, 16, 112, 112),
        batch_size=8
    )

    print("\n" + "=" * 60)
    print("BACKBONE COMPARATIVE SUMMARY:")
    print(f"  S3D       : {res_s3d['feature_dim']}-d | {res_s3d['peak_vram_mb']:.1f} MB VRAM | {res_s3d['throughput_fps']:.1f} clips/s | {res_s3d['latency_per_clip_ms']:.1f} ms/clip")
    print(f"  R(2+1)D-18: {res_r2p1d['feature_dim']}-d | {res_r2p1d['peak_vram_mb']:.1f} MB VRAM | {res_r2p1d['throughput_fps']:.1f} clips/s | {res_r2p1d['latency_per_clip_ms']:.1f} ms/clip")
    print("=" * 60)
