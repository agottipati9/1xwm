import argparse
import glob
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from matplotlib.colors import hsv_to_rgb
from tqdm import tqdm
import os


def load_descriptor_batches(descs_glob: str):
    """Yield (descs[B,1,T,D], meta, path) for each batch file."""
    files = sorted(glob.glob(descs_glob))
    assert files, f"No descriptor files found for pattern: {descs_glob}"
    for f in files:
        pkg = torch.load(f, map_location="cpu", weights_only=False)
        descs = pkg["descriptors"]  # (B,1,T,D')
        meta = pkg["meta"]
        frames = pkg["frames"]
        yield descs, frames, meta, f


def ensure_no_cls(descs: torch.Tensor, n_hw: Tuple[int, int]):
    """
    Ensure we drop CLS and registers if present.
    descs: (B,1,T,D)
    n_hw: (n_h, n_w)
    Returns descs_no_cls: (B,1,T_no_cls,D)
    """
    n_extra = 5 # 1 CLS + 4 Registers
    B, one, T, D = descs.shape
    n_h, n_w = n_hw
    T_expected = n_h * n_w
    if T == T_expected:
        return descs
    elif T == T_expected + 1:
        print('Forgot to remove CLS token from descriptors.')
        return descs[:, :, 1:, :]
    elif T == T_expected + n_extra:
        print('Forgot to remove CLS token and registers from descriptors.')
        return descs[:, :, n_extra:, :]
    else:
        raise ValueError(f"Unexpected token count T={T} vs expected {T_expected} (or {T_expected}+5 with CLS and registers).")


def fit_pca_global(descs_glob: str, batch_chunk: int = 16384) -> Tuple[IncrementalPCA, float, np.ndarray, np.ndarray]:
    """
    Fit a 2D IncrementalPCA across all descriptors for consistent color mapping.
    Also compute global stats needed for normalization:
      - max magnitude of (PC1, PC2)
      - per-component min/max (for optional component-wise normalization)
    Returns: (ipca, max_mag, comp_mins[2], comp_maxs[2])
    """
    ipca = IncrementalPCA(n_components=2)

    # First pass: fit PCA incrementally
    for descs, frames, meta, path in load_descriptor_batches(descs_glob):
        n_h, n_w = tuple(meta["n_patches_hw"])
        descs = ensure_no_cls(descs, (n_h, n_w))
        B, _, T, D = descs.shape
        X = descs.view(B * T, D).numpy()
        # Incremental fit in chunks to limit RAM
        for start in range(0, X.shape[0], batch_chunk):
            ipca.partial_fit(X[start:start + batch_chunk])

    # Second pass: compute global stats on transformed components
    comp_mins = np.array([np.inf, np.inf], dtype=np.float64)
    comp_maxs = np.array([-np.inf, -np.inf], dtype=np.float64)
    max_mag = 0.0

    for descs, frames, meta, path in load_descriptor_batches(descs_glob):
        n_h, n_w = tuple(meta["n_patches_hw"])
        descs = ensure_no_cls(descs, (n_h, n_w))
        B, _, T, D = descs.shape
        X = descs.view(B * T, D).numpy()
        Z = ipca.transform(X)  # (N,2)
        comp_mins = np.minimum(comp_mins, Z.min(axis=0))
        comp_maxs = np.maximum(comp_maxs, Z.max(axis=0))
        mags = np.linalg.norm(Z, axis=1)
        if mags.size:
            max_mag = max(max_mag, float(mags.max()))

    # Avoid degenerate scales
    eps = 1e-8
    comp_maxs = np.maximum(comp_maxs, comp_mins + eps)
    max_mag = max(max_mag, eps)

    return ipca, max_mag, comp_mins, comp_maxs


def pca_to_hsv_rgb(grid_2d: np.ndarray, max_mag: float) -> np.ndarray:
    """
    Map 2D PCA grid (H,W,2) to an HSV image (H,W,3) and then to RGB (float 0..1).
    Hue = angle(pc1, pc2) mapped to [0,1]
    Saturation = 1
    Value = normalized magnitude (0..1)
    """
    pc1 = grid_2d[..., 0]
    pc2 = grid_2d[..., 1]
    angle = np.arctan2(pc2, pc1)  # [-pi, pi]
    hue = (angle + np.pi) / (2 * np.pi)  # [0,1]
    mag = np.sqrt(pc1 ** 2 + pc2 ** 2)
    val = np.clip(mag / max_mag, 0.0, 1.0)
    sat = np.ones_like(val)

    hsv = np.stack([hue, sat, val], axis=-1).astype(np.float32)
    rgb = hsv_to_rgb(hsv)  # float in [0,1]
    return rgb


def overlay_color(frame_bgr: np.ndarray, color_rgb_float: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Upsample color_rgb_float to frame size and alpha blend over BGR frame.
    color_rgb_float: (H_patch, W_patch, 3) in [0,1]
    frame_bgr: (H, W, 3) in uint8
    Returns blended BGR uint8
    """
    H, W = frame_bgr.shape[:2]
    color_rgb_uint8 = (cv2.resize((color_rgb_float * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_CUBIC))
    color_bgr_uint8 = cv2.cvtColor(color_rgb_uint8, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(frame_bgr, 1.0 - alpha, color_bgr_uint8, alpha, 0.0)
    return blended


def visualize(descs_glob: str,
              out_dir: str,
              alpha: float = 0.5,
              write_video: bool = False,
              fps: float = 30.0):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Fit PCA and get global normalization stats
    print("Fitting global 2D PCA...")
    ipca, max_mag, comp_mins, comp_maxs = fit_pca_global(descs_glob)

    writer = None
    try:
        for descs, frames, meta, path in load_descriptor_batches(descs_glob):
            n_h, n_w = tuple(meta["n_patches_hw"])
            frame_indices: List[int] = meta["frame_indices"]

            descs = ensure_no_cls(descs, (n_h, n_w))
            B, _, T, D = descs.shape
            X = descs.view(B * T, D).numpy()
            Z = ipca.transform(X)  # (B*T,2)
            Z = Z.reshape(B, T, 2)

            # Per-frame visualization
            for i in range(B):
                frame_idx = frame_indices[i]
                # Load frame
                frame_bgr = frames[i]
                assert frame_bgr is not None, f"Failed to read frame at index {frame_idx}"

                grid_2d = Z[i].reshape(n_h, n_w, 2)  # (n_h, n_w, 2)
                rgb_feat = pca_to_hsv_rgb(grid_2d, max_mag=max_mag)  # (n_h, n_w, 3) in [0,1]
                overlay = overlay_color(frame_bgr, rgb_feat, alpha=alpha)

                out_img = out_path / f"pca2_overlay_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_img), overlay)

                if write_video:
                    if writer is None:
                        H, W = overlay.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(out_path / "pca2_overlay.mp4"), fourcc, fps, (W, H))
                    writer.write(overlay)
    finally:
        if writer is not None:
            writer.release()

    print(f"Done. Visualizations saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize DINO descriptors per frame using 2D PCA -> HSV color overlay.")
    parser.add_argument("--descs", required=True, type=str,
                        help="Glob pattern for descriptor batch files, e.g., feats/descs_batch*.pt")
    parser.add_argument("--out_dir", required=True, type=str, help="Output directory for overlays/video.")
    parser.add_argument("--alpha", default=0.5, type=float, help="Overlay alpha for blending.")
    parser.add_argument("--write_video", action="store_true", help="Also write an mp4 video.")
    parser.add_argument("--fps", default=30.0, type=float, help="FPS for output video if --write_video.")
    args = parser.parse_args()

    visualize(
        descs_glob=args.descs,
        out_dir=args.out_dir,
        alpha=args.alpha,
        write_video=args.write_video,
        fps=args.fps
    )


if __name__ == "__main__":
    main()