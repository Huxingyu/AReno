# fix(cuda): preserve sampling seeds

CUDA rollout options currently discard `SamplingParams.seed` when constructing
worker sampling parameters. Forward the value, including zero, so an explicitly
seeded SDK request reaches the sampler. The unseeded default remains `None`.

## Validation

On `b84e509`, with CUDA hidden: **5 passed**.

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=. python -m pytest -q tests/test_cuda_generation_cpu.py
```

The added CPU case checks `None`, `0`, and `41` at the options boundary. This
verifies parameter forwarding; it does not establish identical trajectories
across different execution schedules.

## Dependencies

None. The diff contains only `generation.py` and its direct CPU test. Base:
`48d07c5`. Local branch: `review/async-seed`.
