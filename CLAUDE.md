# Un0

See [README.md](README.md) for what this project is, the model recipe, and the
full setup / training / inference / evaluation commands. That is the source of
truth for both readers and contributors.

## Quick reference

```bash
uv sync --group dev    # core + tests + ruff
uv run pytest          # unit tests
uv run ruff check      # lint
```

## Hardware notes

- On Blackwell (sm_100+; B300 reports sm_103), the cuDNN 9.x bundled with
  `torch 2.11+cu128` has no valid SDPA execution plan, so the compiled DINO
  attention crashes with `No valid execution plans built`. Both training entry
  points call `common.disable_cudnn_sdp_on_blackwell()`, which falls back to
  flash, gated on compute capability `>= 10` so pre-Blackwell GPUs (H200, A100)
  keep cuDNN attention. After changing SDPA backends, clear the Inductor cache
  (`/tmp/torchinductor_*`) so a stale compiled graph doesn't keep calling the
  cuDNN attention op.
