"""Build photomosaics from photographs using Un-0 generated tiles.

Each output cell is a generated CIFAR-10 (or ImageNet-64) sample chosen by
average RGB color match. Tiles are generated once and cached for reuse.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image

from un0 import ConditionalImplicitKuramotoGenerator
from un0.apple import (
    configure_apple_runtime,
    default_pretrained_checkpoint,
    generate_samples,
    prepare_model_for_inference,
)
from un0.common import resolve_device, seed_everything
from un0.model import PRETRAINED_NAMES

_DEFAULT_POOL_DIR = Path("samples/.photomosaic_pools")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        type=Path,
        nargs="+",
        help="One or more source photographs (JPEG/PNG).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (single source only). Default: samples/<stem>_mosaic.png",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("samples"),
        help="Directory for outputs when multiple sources are given.",
    )
    parser.add_argument(
        "--pretrained",
        default=None,
        choices=PRETRAINED_NAMES,
        help="Checkpoint for tile generation (default: n4096 on Apple Silicon).",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cols", type=int, default=40, help="Grid columns.")
    parser.add_argument("--rows", type=int, default=0, help="Grid rows (0 = auto from aspect).")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="Cell size in pixels (0 = match model output, usually 32 or 64).",
    )
    parser.add_argument("--pool-size", type=int, default=800, help="Tiles to generate before dedup.")
    parser.add_argument("--preset", choices=("quality", "balanced", "fast"), default="balanced")
    parser.add_argument("--top-k", type=int, default=12, help="Stochastic top-k color matches.")
    parser.add_argument("--reuse-penalty", type=float, default=0.1)
    parser.add_argument("--max-reuse", type=int, default=3, help="Max uses per tile (0 = unlimited).")
    parser.add_argument(
        "--pool-dir",
        type=Path,
        default=_DEFAULT_POOL_DIR,
        help="Directory for cached tile pools.",
    )
    parser.add_argument(
        "--no-cache-pool",
        action="store_true",
        help="Regenerate the tile pool even if a cache file exists.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast preset, smaller pool (400), and at most 32 columns.",
    )
    return parser


def _pool_cache_path(pool_dir: Path, pretrained: str, pool_size: int, preset: str) -> Path:
    safe = pretrained.replace("/", "_")
    return pool_dir / f"{safe}_n{pool_size}_{preset}.pt"


def _dedupe_tiles(tiles: torch.Tensor) -> torch.Tensor:
    mean = tiles.mean(dim=(2, 3))
    codes = (mean * 15.0).round().to(torch.int64)
    keys = codes[:, 0] * 256 + codes[:, 1] * 16 + codes[:, 2]
    keys_np = keys.cpu().numpy()
    _, idx = np.unique(keys_np, return_index=True)
    return tiles[sorted(idx)]


def _flat_to_tiles(flat: torch.Tensor) -> torch.Tensor:
    size = round((flat.shape[1] // 3) ** 0.5)
    tiles = flat.reshape(-1, 3, size, size)
    return ((tiles + 1.0) * 0.5).clamp(0.0, 1.0).cpu()


def _load_or_build_pool(
    model: ConditionalImplicitKuramotoGenerator,
    device: torch.device,
    *,
    pool_size: int,
    preset: str,
    cache_path: Path,
    use_cache: bool,
) -> torch.Tensor:
    if use_cache and cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        tiles = payload["tiles"].clamp(0.0, 1.0)
        print(f"Loaded {tiles.shape[0]} cached tiles from {cache_path}")
        return tiles

    num_classes = int(model.dynamics.num_classes)
    class_ids = torch.arange(pool_size, device=device, dtype=torch.long) % num_classes
    t0 = time.perf_counter()
    flat = generate_samples(model, class_ids, device, preset=preset)  # type: ignore[arg-type]
    tiles = _dedupe_tiles(_flat_to_tiles(flat))
    print(f"Generated {tiles.shape[0]} unique tiles in {time.perf_counter() - t0:.1f}s")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tiles": tiles, "pretrained": cache_path.stem}, cache_path)
    print(f"Cached tile pool to {cache_path}")
    return tiles


def _source_grid(
    source: Path,
    cols: int,
    rows: int,
    tile_size: int,
) -> tuple[np.ndarray, int, int]:
    img = Image.open(source).convert("RGB")
    if rows <= 0:
        aspect = img.height / img.width
        rows = max(1, round(cols * aspect))
    target_w = cols * tile_size
    target_h = rows * tile_size
    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    cells = arr.reshape(rows, tile_size, cols, tile_size, 3).mean(axis=(1, 3))
    return cells, cols, rows


def _pick_tile_index(
    dist: torch.Tensor,
    usage: torch.Tensor,
    *,
    top_k: int,
    reuse_penalty: float,
    max_reuse: int,
    generator: torch.Generator,
) -> int:
    scored = dist + reuse_penalty * usage
    if max_reuse > 0:
        scored = scored.masked_fill(usage >= max_reuse, float("inf"))

    k = min(int(top_k), int(scored.shape[0]))
    if not torch.isfinite(scored).any():
        return int(dist.argmin().item())

    _, candidates = torch.topk(scored, k, largest=False)
    candidate_dist = scored[candidates].clamp_min(1e-8)
    weights = 1.0 / candidate_dist
    weights = weights / weights.sum()
    pick = int(torch.multinomial(weights, 1, generator=generator).item())
    return int(candidates[pick].item())


def _assemble_mosaic(
    cells: np.ndarray,
    tiles: torch.Tensor,
    *,
    top_k: int,
    reuse_penalty: float,
    max_reuse: int,
    seed: int,
) -> torch.Tensor:
    rows, cols, _ = cells.shape
    tile_means = tiles.mean(dim=(2, 3))
    cell_tensor = torch.from_numpy(cells).to(tile_means.dtype)
    cell_flat = cell_tensor.reshape(-1, 3)
    dist = torch.cdist(cell_flat, tile_means)

    usage = torch.zeros(tiles.shape[0], dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    indices: list[int] = []
    for row_dist in dist.reshape(rows * cols, -1):
        idx = _pick_tile_index(
            row_dist,
            usage,
            top_k=top_k,
            reuse_penalty=reuse_penalty,
            max_reuse=max_reuse,
            generator=generator,
        )
        usage[idx] += 1.0
        indices.append(idx)

    tile_h, tile_w = int(tiles.shape[2]), int(tiles.shape[3])
    chosen = tiles[indices].reshape(rows, cols, 3, tile_h, tile_w)
    mosaic = chosen.permute(0, 3, 1, 4, 2).reshape(rows * tile_h, cols * tile_w, 3)
    return mosaic.permute(2, 0, 1)


def _resolve_output_path(
    source: Path,
    *,
    output: Path | None,
    output_dir: Path,
    multiple_sources: bool,
) -> Path:
    if output is not None and not multiple_sources:
        return output
    return output_dir / f"{source.stem}_mosaic.png"


def _render_mosaic(
    source: Path,
    output: Path,
    tiles: torch.Tensor,
    *,
    cols: int,
    rows: int,
    tile_size: int,
    top_k: int,
    reuse_penalty: float,
    max_reuse: int,
    seed: int,
) -> None:
    if not source.is_file():
        msg = f"Source image not found: {source}"
        raise FileNotFoundError(msg)

    print(f"Reading {source}...")
    cells, grid_cols, grid_rows = _source_grid(source, cols, rows, tile_size)
    print(f"Grid: {grid_cols}x{grid_rows} = {grid_cols * grid_rows} cells")

    t0 = time.perf_counter()
    mosaic = _assemble_mosaic(
        cells,
        tiles,
        top_k=top_k,
        reuse_penalty=reuse_penalty,
        max_reuse=max_reuse,
        seed=seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(mosaic, output)
    h, w = mosaic.shape[1], mosaic.shape[2]
    print(f"Wrote {output} ({w}x{h}) in {time.perf_counter() - t0:.1f}s")


def main() -> None:
    args = build_parser().parse_args()
    if args.quick:
        args.preset = "fast"
        args.pool_size = min(int(args.pool_size), 400)
        if int(args.cols) > 32:
            args.cols = 32

    if args.output is not None and len(args.sources) > 1:
        msg = "--output accepts a single path; use --output-dir for multiple sources."
        raise SystemExit(msg)

    configure_apple_runtime()
    seed_everything(int(args.seed))
    device = resolve_device(str(args.device))

    pretrained = args.pretrained or default_pretrained_checkpoint()
    print(f"Loading {pretrained} on {device}...")
    model = ConditionalImplicitKuramotoGenerator.from_pretrained(pretrained, device=device)
    prepare_model_for_inference(model)

    tile_size = int(args.tile_size)
    if tile_size <= 0:
        probe = model.sample(torch.tensor([0], device=device, dtype=torch.long))
        tile_size = round((probe.shape[1] // 3) ** 0.5)

    cache_path = _pool_cache_path(
        Path(args.pool_dir),
        pretrained,
        int(args.pool_size),
        str(args.preset),
    )
    tiles = _load_or_build_pool(
        model,
        device,
        pool_size=int(args.pool_size),
        preset=str(args.preset),
        cache_path=cache_path,
        use_cache=not bool(args.no_cache_pool),
    )

    multiple = len(args.sources) > 1
    for source in args.sources:
        out = _resolve_output_path(
            source,
            output=args.output,
            output_dir=Path(args.output_dir),
            multiple_sources=multiple,
        )
        _render_mosaic(
            source,
            out,
            tiles,
            cols=int(args.cols),
            rows=int(args.rows),
            tile_size=tile_size,
            top_k=int(args.top_k),
            reuse_penalty=float(args.reuse_penalty),
            max_reuse=int(args.max_reuse),
            seed=int(args.seed),
        )


if __name__ == "__main__":
    main()
