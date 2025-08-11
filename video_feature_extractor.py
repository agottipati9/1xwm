import argparse
import cv2
import torch
from pathlib import Path
from typing import List, Dict, Any
from misc_utils.extractor import ViTExtractor
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import time

def frame_to_tensor(img_bgr, load_size, mean, std):
    # Convert BGR (OpenCV) to RGB PIL, resize, normalize like extractor.preprocess
    pil_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    # if load_size is not None:

    #     pil = pil.resize((load_size, load_size), Image.LANCZOS) if isinstance(load_size, int) else pil.resize(load_size, Image.LANCZOS)
    # arr = np.asarray(pil).astype("float32") / 255.0
    # tensor = torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W
    # # Normalize
    # mean_t = torch.tensor(mean)[:, None, None]
    # std_t = torch.tensor(std)[:, None, None]
    # tensor = (tensor - mean_t) / std_t
    # return tensor, pil.size  # (W,H)
    if load_size is not None:
        pil_image = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)
    prep = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    prep_img = prep(pil_image) # [None, ...]
    return prep_img, pil_image.size

def save_batch(descriptors: torch.Tensor,
               frames: List[cv2.typing.MatLike],
               frame_indices: List[int],
               n_patches_hw,
               frame_sizes: List[tuple],
               out_dir: Path,
               prefix: str,
               batch_id: int):
    """
    descriptors: shape (B, 1, T, D')
    frames: list of raw video frames
    n_patches_hw: (n_h, n_w)
    frame_sizes: list of (W,H) after resize
    """
    meta = {
        "frame_indices": frame_indices,
        "n_patches_hw": n_patches_hw,
        "frame_sizes": frame_sizes,
        "descriptor_shape": list(descriptors.shape),
        "time_saved": time.time()
    }
    torch.save(
        {
            "descriptors": descriptors.cpu(),
            "frames": frames,
            "meta": meta
        },
        out_dir / f"{prefix}_batch{batch_id:06d}.pt"
    )
    with open(out_dir / f"{prefix}_batch{batch_id:06d}.json", "w") as f:
        json.dump(meta, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Extract DINO descriptors from a video.")
    parser.add_argument("--video_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--model_type", default="dino_vitb8", type=str)
    parser.add_argument("--stride", default=4, type=int)
    parser.add_argument("--load_size", default=224, type=int, help="Resize frames to square load_size. Use same as training size for consistency.")
    parser.add_argument("--facet", default="token", type=str, choices=["key","query","value","token"])
    parser.add_argument("--layer", default=11, type=int)
    parser.add_argument("--bin", action="store_true")
    parser.add_argument("--include_cls", action="store_true", help="Include CLS token in descriptors (ignored if --bin).")
    parser.add_argument("--frame_stride", default=1, type=int, help="Sample every Nth frame.")
    parser.add_argument("--max_frames", default=-1, type=int, help="Limit number of frames (-1 for all).")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--prefix", default="descs", type=str, help="Output file prefix.")
    parser.add_argument("--saliency", action="store_true", help="Also extract saliency maps (only dino_vits8).")
    parser.add_argument("--fp16", action="store_true", help="Use half precision autocast.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = ViTExtractor(model_type=args.model_type, stride=args.stride, device=device)
    mean, std = extractor.mean, extractor.std  # from initialized extractor

    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), f"Cannot open video: {args.video_path}"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    print(f"Opened video with {total_frames} frames (reported)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_tensors = []
    batch_indices = []
    batch_frame_sizes = []
    batch_frames = [] # For debugging
    batch_id = 0
    global_count = 0
    sampled_count = 0

    autocast_ctx = torch.amp.autocast("cuda") if (args.fp16 and device == "cuda") else torch.amp.autocast("cpu")
    # (If CPU + fp16 requested, it will silently do nothing.)

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if global_count % args.frame_stride != 0:
                global_count += 1
                continue

            tensor, frame_size = frame_to_tensor(frame, args.load_size, mean, std)
            batch_tensors.append(tensor)
            batch_indices.append(global_count)
            batch_frame_sizes.append(frame_size)
            batch_frames.append(frame)
            sampled_count += 1

            if len(batch_tensors) == args.batch_size:
                # Process batch
                imgs = torch.stack(batch_tensors, dim=0).to(device)  # B,C,H,W
                with autocast_ctx:
                    descs = extractor.extract_descriptors(
                        imgs,
                        layer=args.layer,
                        facet=args.facet,
                        bin=args.bin,
                        include_cls=args.include_cls
                    )  # shape B x 1 x T x D'

                    save_batch(descs, batch_frames, batch_indices, extractor.num_patches, batch_frame_sizes, out_dir, args.prefix, batch_id)

                    if args.saliency:
                        sal = extractor.extract_saliency_maps(imgs)  # B x (T_no_cls)
                        torch.save(
                            {
                                "saliency": sal.cpu(),
                                "frame_indices": batch_indices,
                                "n_patches_hw": extractor.num_patches
                            },
                            out_dir / f"{args.prefix}_saliency_batch{batch_id:06d}.pt"
                        )

                batch_id += 1
                batch_tensors.clear()
                batch_indices.clear()
                batch_frame_sizes.clear()
                batch_frames.clear()

            global_count += 1
            if 0 < args.max_frames <= sampled_count:
                print("Reached max_frames limit.")
                break

        # Flush remainder
        if batch_tensors:
            imgs = torch.stack(batch_tensors, dim=0).to(device)
            with autocast_ctx:
                descs = extractor.extract_descriptors(
                    imgs,
                    layer=args.layer,
                    facet=args.facet,
                    bin=args.bin,
                    include_cls=args.include_cls
                )
                save_batch(descs, batch_frames, batch_indices, extractor.num_patches, batch_frame_sizes, out_dir, args.prefix, batch_id)
                if args.saliency:
                    sal = extractor.extract_saliency_maps(imgs)
                    torch.save(
                        {
                            "saliency": sal.cpu(),
                            "frame_indices": batch_indices,
                            "n_patches_hw": extractor.num_patches
                        },
                        out_dir / f"{args.prefix}_saliency_batch{batch_id:06d}.pt"
                    )
    cap.release()
    print(f"Done. Sampled frames: {sampled_count}. Batches saved: {batch_id + (1 if batch_tensors else 0)}")

if __name__ == "__main__":
    main()