import argparse
import logging
import math
import os
import os.path as osp
import time
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

LOGGER = logging.getLogger("torchscript_infer")


def nms_seeds_tiny(seeds: Sequence[dict], thr: float) -> List[dict]:
    """Apply tiny endpoint NMS for lane seeds."""

    def cal_dis(p1: Sequence[float], p2: Sequence[float]) -> float:
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def search_groups(coord, groups, local_thr):
        for idx_group, group in enumerate(groups):
            for group_point in group:
                if cal_dis(coord, group_point[1]) <= local_thr:
                    return idx_group
        return -1

    groups = []
    for idx, (coord, score) in enumerate([(s["coord"], s["score"]) for s in seeds]):
        idx_group = search_groups(coord, groups, thr)
        if idx_group < 0:
            groups.append([(idx, coord, score)])
        else:
            groups[idx_group].append((idx, coord, score))
    keep_idxes = [max(group, key=lambda x: x[2])[0] for group in groups]
    return [seeds[idx] for idx in keep_idxes]


def compute_locations(shape: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """Create row coordinate map used by lane post processing."""
    pos = torch.arange(0, shape[-1], step=1, dtype=torch.float32, device=device)
    pos = pos.reshape((1, 1, -1))
    return pos.repeat(shape[0], shape[1], 1)


class CurvelanesPostProcessor:
    """Minimal Curvelanes post-processor with no external dependencies."""

    def __init__(self, mask_size: Tuple[int, int, int], device: torch.device,
                 use_offset: bool = True):
        self.use_offset = use_offset
        self.pos = compute_locations(mask_size, device=device).repeat(100, 1, 1)

    def lane_post_process_all(self, seeds: Sequence[dict], downscale: int) -> List[dict]:
        masks, regs, ranges, scores, seed_idxes = [], [], [], [], []
        for idx, seed in enumerate(seeds):
            masks.append(seed["mask"])
            regs.append(seed["reg"])
            ranges.append(seed["range"])
            for _ in range(seed["mask"].size(0)):
                scores.append(seed["score"])
                seed_idxes.append(idx)

        if not scores:
            return []

        masks = torch.cat(masks, 0)
        regs = torch.cat(regs, 0)
        ranges = torch.cat(ranges, 0)
        num_ins = masks.size(0)

        mask_softmax = F.softmax(masks, dim=-1)
        row_pos = torch.sum(self.pos[:num_ins] * mask_softmax, dim=2).detach().cpu().numpy().astype(
            np.int32)
        ranges_np = torch.argmax(ranges, 1).detach().cpu().numpy()
        regs_np = regs.detach().cpu().numpy()

        lane_ends = []
        max_rows = ranges_np.shape[1]
        for lane_range in ranges_np:
            min_idx = max_idx = None
            for row_idx, valid in enumerate(lane_range):
                if valid:
                    min_idx = max(0, row_idx - 1)
                    break
            for row_idx, valid in enumerate(lane_range[::-1]):
                if valid:
                    max_idx = min(max_rows - 1, len(lane_range) - row_idx)
                    break
            lane_ends.append([min_idx, max_idx])

        lanes = []
        for lane_idx, (y0, y1) in enumerate(lane_ends):
            if y0 is None or y1 is None:
                continue
            selected_ys = np.arange(y0, y1 + 1)
            selected_xs = row_pos[lane_idx, :][selected_ys]
            if self.use_offset:
                selected_regs = regs_np[lane_idx, selected_ys, selected_xs]
            else:
                selected_regs = 0.5
            points = np.concatenate(
                (np.expand_dims(selected_xs, 1), np.expand_dims(selected_ys, 1)),
                1).astype(np.float32)
            points[:, 0] = points[:, 0] + selected_regs
            points *= downscale
            if len(points) > 1:
                lanes.append(
                    dict(points=points, score=scores[lane_idx], seed=seeds[seed_idxes[lane_idx]]))
        return lanes

    def __call__(self, output: Sequence[dict], downscale: int) -> Tuple[List[dict], List[dict]]:
        seeds = nms_seeds_tiny(output, 1)
        lanes = self.lane_post_process_all(seeds, downscale)
        return lanes, seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TorchScript inference on an image directory")
    parser.add_argument("model", help="Path to TorchScript model (.pt/.torchscript)")
    parser.add_argument("image_dir", help="Directory containing input images")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Inference device, e.g. cuda:0 or cpu")
    parser.add_argument("--width", type=int, default=800, help="Resize width for model input")
    parser.add_argument("--height", type=int, default=416, help="Resize height for model input")
    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=[75.3, 76.6, 77.6],
        help="Normalization mean (BGR order)")
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=[50.5, 53.8, 54.3],
        help="Normalization std (BGR order)")
    parser.add_argument("--to-rgb", action="store_true", help="Convert BGR to RGB")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--save-dir", default=None, help="Directory to save visualization images")
    parser.add_argument(
        "--mask-size",
        type=int,
        nargs=3,
        default=[1, 40, 100],
        help="Mask size used by lane post-processor")
    parser.add_argument("--down-scale", type=int, default=8, help="Lane output downscale")
    parser.add_argument(
        "--exts",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
        help="Image extensions to include")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level), format="%(asctime)s | %(levelname)s | %(message)s")


def list_images(image_dir: str, exts: Sequence[str]) -> List[Path]:
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    ext_set = {e.lower() for e in exts}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ext_set]
    files.sort()
    if not files:
        raise ValueError(f"No images found in {image_dir} with extensions: {sorted(ext_set)}")
    return files


def preprocess_image(image_path: Path, input_size: Tuple[int, int], mean: np.ndarray,
                     std: np.ndarray, to_rgb: bool) -> torch.Tensor:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    img = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def describe_output(output: Any) -> str:
    if torch.is_tensor(output):
        return f"Tensor{tuple(output.shape)}"
    if isinstance(output, (list, tuple)):
        parts = [describe_output(item) for item in output[:3]]
        suffix = ", ..." if len(output) > 3 else ""
        return f"{type(output).__name__}[{', '.join(parts)}{suffix}]"
    if isinstance(output, dict):
        keys = list(output.keys())
        return f"dict(keys={keys[:5]}{'...' if len(keys) > 5 else ''})"
    return type(output).__name__


def _is_lane_seed_output(output: Any) -> bool:
    if not isinstance(output, (list, tuple)) or len(output) < 1:
        return False
    seeds = output[0]
    if not isinstance(seeds, list):
        return False
    return len(seeds) == 0 or isinstance(seeds[0], dict)


def _draw_lanes_on_image(image: np.ndarray, lanes: Sequence[dict],
                         input_size: Tuple[int, int]) -> np.ndarray:
    vis = image.copy()
    ori_h, ori_w = vis.shape[:2]
    in_w, in_h = input_size
    sx = float(ori_w) / max(float(in_w), 1.0)
    sy = float(ori_h) / max(float(in_h), 1.0)
    for lane in lanes:
        points = lane.get("points")
        if points is None or len(points) < 2:
            continue
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0] * sx, 0, ori_w - 1)
        pts[:, 1] = np.clip(pts[:, 1] * sy, 0, ori_h - 1)
        cv2.polylines(vis, [np.round(pts).astype(np.int32)], False, (0, 255, 0), 2)
    return vis


def _build_save_path(save_dir: str, image_root: str, image_path: Path) -> str:
    return osp.join(save_dir, osp.relpath(str(image_path), image_root))


def run_inference(model: torch.jit.ScriptModule, image_paths: Sequence[Path], image_root: str,
                  device: torch.device, input_size: Tuple[int, int], mean: np.ndarray,
                  std: np.ndarray, to_rgb: bool, warmup: int, save_dir: str,
                  mask_size: Tuple[int, int, int], down_scale: int) -> None:
    sample = preprocess_image(image_paths[0], input_size, mean, std, to_rgb).to(device)
    post_processor = CurvelanesPostProcessor(mask_size=mask_size, device=device)
    warned_non_lane = False

    inference_ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    with inference_ctx():
        for _ in range(max(warmup, 0)):
            _ = model(sample)
        sync_if_cuda(device)

        latencies_ms: List[float] = []
        output_signature = ""
        start_total = time.perf_counter()
        for idx, img_path in enumerate(image_paths):
            inp = preprocess_image(img_path, input_size, mean, std, to_rgb).to(device)
            sync_if_cuda(device)
            t0 = time.perf_counter()
            out = model(inp)
            sync_if_cuda(device)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            if idx == 0:
                output_signature = describe_output(out)

            if save_dir is not None:
                out_file = _build_save_path(save_dir, image_root, img_path)
                os.makedirs(osp.dirname(out_file), exist_ok=True)
                image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise ValueError(f"Failed to read image for save: {img_path}")
                if _is_lane_seed_output(out):
                    lanes, _ = post_processor(out[0], down_scale)
                    vis_img = _draw_lanes_on_image(image_bgr, lanes, input_size)
                    cv2.imwrite(out_file, vis_img)
                else:
                    if not warned_non_lane:
                        LOGGER.warning(
                            "Model output is not lane seed format. Saving resized inputs.")
                        warned_non_lane = True
                    cv2.imwrite(
                        out_file,
                        cv2.resize(image_bgr, input_size, interpolation=cv2.INTER_LINEAR),
                    )
        total_s = time.perf_counter() - start_total

    fps = len(image_paths) / max(total_s, 1e-8)
    lat_arr = np.asarray(latencies_ms, dtype=np.float64)
    LOGGER.info("Images processed: %d", len(image_paths))
    LOGGER.info("Input size (W,H): %s", input_size)
    LOGGER.info("Model output: %s", output_signature)
    LOGGER.info("Total time: %.4f s", total_s)
    LOGGER.info("FPS: %.2f", fps)
    if save_dir is not None:
        LOGGER.info("Saved visualizations to: %s", save_dir)
    LOGGER.info(
        "Latency (ms) | mean: %.3f | p50: %.3f | p90: %.3f | p99: %.3f | max: %.3f",
        float(np.mean(lat_arr)),
        float(np.percentile(lat_arr, 50)),
        float(np.percentile(lat_arr, 90)),
        float(np.percentile(lat_arr, 99)),
        float(np.max(lat_arr)),
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"TorchScript model not found: {args.model}")
    if not osp.isdir(args.image_dir):
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    image_paths = list_images(args.image_dir, args.exts)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    LOGGER.info("Loading TorchScript model: %s", model_path)
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    model.to(device)

    mean = np.asarray(args.mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(args.std, dtype=np.float32).reshape(1, 1, 3)
    input_size = (args.width, args.height)
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    run_inference(
        model=model,
        image_paths=image_paths,
        image_root=args.image_dir,
        device=device,
        input_size=input_size,
        mean=mean,
        std=std,
        to_rgb=args.to_rgb,
        warmup=args.warmup,
        save_dir=args.save_dir,
        mask_size=tuple(args.mask_size),
        down_scale=args.down_scale,
    )


if __name__ == "__main__":
    main()
