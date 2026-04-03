# ML Performance — Code & Model Optimization

Apply these standards when writing inference, embedding extraction, or search code.

## Batch Processing
- Never embed one image at a time — minimum batch size 32, tune based on available memory
- Use `torch.utils.data.DataLoader` for batched image loading, but choose `num_workers` based on the actual workload and platform — `0` is valid when multiprocessing overhead outweighs the benefit
- Prefer `pin_memory=True` for CUDA DataLoaders
- Validate loader-related CLI/config inputs explicitly (`num_workers >= 0`, `prefetch_factor >= 1`, invalid combinations rejected up front)
- Do not assume training and validation/test loaders should share the same settings; evaluation loaders should usually use fewer workers and more conservative resource usage
- Only enable `persistent_workers=True` for long-lived DataLoaders reused across epochs; do not use it on short-lived or repeatedly recreated loaders
- Make high-impact knobs such as `num_workers`, `prefetch_factor`, `pin_memory`, and `persistent_workers` configurable rather than hardcoding aggressive defaults

## Inference Mode
- Use `torch.inference_mode()` for all evaluation and embedding extraction (faster than `torch.no_grad()`)
- Use `torch.compile()` where supported and beneficial — profile first
- Disable gradient computation and dropout explicitly

## Device Management
- Prefer CUDA when available, then MPS on macOS, then CPU:
  ```python
  if torch.cuda.is_available():
      device = torch.device("cuda")
  elif torch.backends.mps.is_available():
      device = torch.device("mps")
  else:
      device = torch.device("cpu")
  ```
- Fall back to CPU gracefully — never crash if CUDA/MPS is unavailable
- Minimize data copies between CPU and MPS/GPU; keep tensors on device through the pipeline
- Enable `torch.backends.cudnn.benchmark = True` only for fixed-size CUDA workloads, and log/document the reproducibility tradeoff when doing so
- Be aware of MPS limitations: some ops may fall back to CPU silently

## Memory Optimization
- Prefer `float16` for inference — matches CoreML deployment precision
- Stream large datasets with `IterableDataset` or memory-mapped arrays when they don't fit in RAM
- Pre-compute and cache gallery embeddings to disk as numpy `.npy` or memory-mapped files — avoid re-embedding on every evaluation run
- Clear GPU/MPS cache between large batch operations with `torch.mps.empty_cache()`

## Image Loading
- Use `torchvision.io` or `PIL` with lazy loading
- Pre-resize large images (>1024px) before model input to avoid OOM
- For gallery embedding: resize once and cache, don't resize on every load

## Profiling
- Profile before optimizing: `torch.profiler` for MPS/GPU paths, `cProfile` for CPU-bound code
- Benchmark with realistic batch sizes matching deployment constraints:
  - Single-image inference for on-device simulation
  - Batched inference for offline evaluation
- Label benchmark results clearly:
  - apples-to-apples = same model, batch size, worker settings, and input set
  - best-stable practical = configs differ to maximize stable throughput on each device
- Do not present mixed-config measurements as pure hardware comparisons
- Report p50/p95/p99 latencies, not just mean

## Search Performance
- For brute-force cosine similarity: use numpy vectorized operations or `torch.mm`
- Pre-normalize embeddings to unit length so cosine similarity = dot product
- Use float16 for index storage; cast to float32 only for the search computation if needed
