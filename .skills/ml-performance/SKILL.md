# ML Performance — Code & Model Optimization

Apply these standards when writing inference, embedding extraction, or search code.

## Batch Processing
- Never embed one image at a time — minimum batch size 32, tune based on available memory
- Use `torch.utils.data.DataLoader` with `num_workers` > 0 for I/O-bound image loading
- Prefer `pin_memory=True` when using GPU/MPS

## Inference Mode
- Use `torch.inference_mode()` for all evaluation and embedding extraction (faster than `torch.no_grad()`)
- Use `torch.compile()` where supported and beneficial — profile first
- Disable gradient computation and dropout explicitly

## Device Management
- Use MPS (Metal Performance Shaders) on macOS when available:
  ```python
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  ```
- Fall back to CPU gracefully — never crash if MPS is unavailable
- Minimize data copies between CPU and MPS/GPU; keep tensors on device through the pipeline
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
- Report p50/p95/p99 latencies, not just mean

## Search Performance
- For brute-force cosine similarity: use numpy vectorized operations or `torch.mm`
- Pre-normalize embeddings to unit length so cosine similarity = dot product
- Use float16 for index storage; cast to float32 only for the search computation if needed
