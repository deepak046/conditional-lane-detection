import argparse
import os.path as osp
import sys
from typing import Sequence, Tuple

import mmcv
import torch
from mmcv.runner import load_checkpoint

# Ensure local package imports resolve when running as `python tools/*.py`.
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mmdet.models import build_detector
from mmdet.ops import RoIAlign, RoIPool


def _parse_input_shape(shape: Sequence[int]) -> Tuple[int, int, int]:
    """Convert CLI shape argument into CHW input shape."""
    if len(shape) == 1:
        return 3, int(shape[0]), int(shape[0])
    if len(shape) == 2:
        return 3, int(shape[0]), int(shape[1])
    raise ValueError('Invalid input shape. Use --shape H W or --shape S.')


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='MMDet PyTorch model conversion to TorchScript')
    parser.add_argument('config', help='Test config file path')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Output TorchScript filename (.pt or .torchscript)')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='Input image size')
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Run torch.jit.optimize_for_inference before saving')
    return parser.parse_args()


def build_export_model(config_path: str, checkpoint_path: str) -> torch.nn.Module:
    """Build and prepare a detector for TorchScript export."""
    cfg = mmcv.Config.fromfile(config_path)
    cfg.model.pretrained = None

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.cpu().eval()

    # Keep export behavior consistent with ONNX utility.
    for module in model.modules():
        if isinstance(module, (RoIPool, RoIAlign)):
            module.use_torchvision = True

    if not hasattr(model, 'forward_dummy'):
        raise NotImplementedError(
            'TorchScript conversion is not supported with '
            f'{model.__class__.__name__}: missing forward_dummy')
    model.forward = model.forward_dummy
    return model


def export_torchscript_model(model: torch.nn.Module, input_shape: Tuple[int, int,
                                                                        int],
                             optimize: bool) -> torch.jit.ScriptModule:
    """Trace model with dummy input and return TorchScript module."""
    input_tensor = torch.empty(
        (1, *input_shape),
        dtype=next(model.parameters()).dtype,
        device=next(model.parameters()).device)

    with torch.no_grad():
        ts_model = torch.jit.trace(model, input_tensor, strict=False)
        ts_model = torch.jit.freeze(ts_model)
        if optimize:
            ts_model = torch.jit.optimize_for_inference(ts_model)
    return ts_model


def main() -> None:
    args = parse_args()
    if not (args.out.endswith('.pt') or args.out.endswith('.torchscript')):
        raise ValueError('Output file must end with .pt or .torchscript.')

    input_shape = _parse_input_shape(args.shape)
    model = build_export_model(args.config, args.checkpoint)
    ts_model = export_torchscript_model(model, input_shape, args.optimize)
    ts_model.save(args.out)
    print(f'Saved TorchScript model to {args.out}')


if __name__ == '__main__':
    main()
