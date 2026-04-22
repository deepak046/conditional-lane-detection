import warnings
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from mmdet.ops import RoIAlign, RoIPool


def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def _to_cpu_result(data):
    """Recursively move tensor results to CPU for visualization."""
    if torch.is_tensor(data):
        return data.detach().cpu()
    if isinstance(data, list):
        return [_to_cpu_result(item) for item in data]
    if isinstance(data, tuple):
        return tuple(_to_cpu_result(item) for item in data)
    if isinstance(data, dict):
        return {key: _to_cpu_result(value) for key, value in data.items()}
    return data


def _is_lane_result(result):
    return (isinstance(result, (list, tuple)) and len(result) >= 1
            and isinstance(result[0], list)
            and (len(result[0]) == 0 or isinstance(result[0][0], dict)))


def _infer_test_image_shape(cfg, fallback_shape):
    """Infer test-time image shape (h, w) from pipeline config."""
    pipeline = getattr(getattr(cfg, 'data', None), 'test', None)
    pipeline = getattr(pipeline, 'pipeline', None)
    if pipeline is None:
        return fallback_shape

    for transform in pipeline:
        if not isinstance(transform, dict):
            continue
        t = transform.get('type')
        if t == 'Resize' and 'img_scale' in transform:
            img_scale = transform['img_scale']
            if isinstance(img_scale, (list, tuple)) and len(img_scale) >= 2:
                return int(img_scale[1]), int(img_scale[0])
        if t == 'albumentation':
            for step in transform.get('pipelines', []):
                if not isinstance(step, dict):
                    continue
                if step.get('type') == 'Resize':
                    h = step.get('height')
                    w = step.get('width')
                    if h is not None and w is not None:
                        return int(h), int(w)
    return fallback_shape


def _draw_lane_result(model,
                      img,
                      result,
                      score_thr=0.3,
                      rescale_points=True,
                      img_meta=None):
    """Render CondLaneNet-style lane outputs into an image."""
    seeds = result[0]
    if not seeds:
        return mmcv.imread(img)

    cfg = getattr(model, 'cfg', None)
    mask_size = tuple(getattr(cfg, 'mask_size', (1, 40, 100)))
    down_scale = 4
    if cfg is not None and getattr(cfg, 'test_cfg', None) is not None:
        down_scale = cfg.test_cfg.get('out_scale', down_scale)

    model_name = model.__class__.__name__.lower()
    if 'curve' in model_name or 'rnn' in model_name:
        from tools.condlanenet.post_process import CurvelanesPostProcessor
        post_processor = CurvelanesPostProcessor(mask_size=mask_size)
    else:
        from mmdet.models.detectors.condlanenet import CondLanePostProcessor
        post_processor = CondLanePostProcessor(mask_size=mask_size)

    lanes, _ = post_processor(seeds, down_scale)
    drawn = mmcv.imread(img).copy()
    ori_h, ori_w = drawn.shape[:2]
    if rescale_points:
        if cfg is not None:
            net_h, net_w = _infer_test_image_shape(cfg, (ori_h, ori_w))
        else:
            net_h, net_w = ori_h, ori_w
        scale_x = float(ori_w) / max(float(net_w), 1.0)
        scale_y = float(ori_h) / max(float(net_h), 1.0)
    else:
        scale_x = 1.0
        scale_y = 1.0

    # Curvelanes-style datasets keep predictions in resized/cropped coordinates.
    # Project them back to original image space when meta is available.
    crop_offset = crop_shape = proc_img_shape = None
    if isinstance(img_meta, dict):
        crop_offset = img_meta.get('crop_offset')
        crop_shape = img_meta.get('crop_shape')
        proc_img_shape = img_meta.get('img_shape')
    has_crop_projection = (
        crop_offset is not None and crop_shape is not None
        and proc_img_shape is not None)

    if has_crop_projection:
        proc_h, proc_w = proc_img_shape[:2]
        ratio_x = float(crop_shape[1]) / max(float(proc_w), 1.0)
        ratio_y = float(crop_shape[0]) / max(float(proc_h), 1.0)
        offset_x, offset_y = float(crop_offset[0]), float(crop_offset[1])

    for lane in lanes:
        if float(lane.get('score', 0.0)) < score_thr:
            continue
        points = lane.get('points')
        if points is None or len(points) < 2:
            continue
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if has_crop_projection:
            points[:, 0] = points[:, 0] * ratio_x + offset_x
            points[:, 1] = points[:, 1] * ratio_y + offset_y
        else:
            points[:, 0] *= scale_x
            points[:, 1] *= scale_y
        points[:, 0] = np.clip(points[:, 0], 0, ori_w - 1)
        points[:, 1] = np.clip(points[:, 1], 0, ori_h - 1)
        points = np.round(points).astype(np.int32)
        cv2.polylines(drawn, [points], False, (0, 255, 0), thickness=2)
    return drawn


def _build_inference_pipeline(cfg):
    """Build a test pipeline that does not require ground truth fields."""
    lane_target_collectors = {'CollectLane', 'CollectRNNLanes'}
    image_loaders = {'LoadImageFromFile', 'LoadImageFromWebcam'}
    safe_meta_keys = ('filename', 'ori_shape', 'img_shape', 'img_norm_cfg')
    pipeline_cfg = deepcopy(cfg.data.test.pipeline)
    test_pipeline = [LoadImage()]

    replaced_lane_collector = False
    has_collect = False
    for transform in pipeline_cfg:
        if not isinstance(transform, dict):
            test_pipeline.append(transform)
            continue

        transform_type = transform.get('type')
        if transform_type in image_loaders:
            continue
        if transform_type == 'albumentation':
            # Inference has no GT annotations; disable target-aware albumentation.
            transform = deepcopy(transform)
            pipelines = transform.get('pipelines', [])
            for pipeline_step in pipelines:
                if not isinstance(pipeline_step, dict):
                    continue
                if pipeline_step.get('type') != 'Compose':
                    continue
                compose_params = pipeline_step.get('params')
                if isinstance(compose_params, dict):
                    compose_params['bboxes'] = False
                    compose_params['keypoints'] = False
                    compose_params['masks'] = False

        if transform_type in lane_target_collectors:
            replaced_lane_collector = True
            continue
        if transform_type == 'Collect':
            has_collect = True
            # Force Collect to request only inference-time fields.
            transform = deepcopy(transform)
            transform['keys'] = ['img']
            transform['meta_keys'] = safe_meta_keys
        test_pipeline.append(transform)

    if replaced_lane_collector and not has_collect:
        test_pipeline.append(
            dict(type='Collect', keys=['img'], meta_keys=safe_meta_keys))

    return Compose(test_pipeline)


def inference_detector(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = _build_inference_pipeline(cfg)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        device_id = device.index if device.index is not None else 0
        data = scatter(data, [device_id])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


async def async_inference_detector(model, img):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        Awaitable detection results.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = _build_inference_pipeline(cfg)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    device_id = device.index if device.index is not None else 0
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    result = await model.aforward_test(rescale=True, **data)
    return result


def show_result_pyplot(model,
                       img,
                       result,
                       score_thr=0.3,
                       fig_size=(15, 10),
                       show=True):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
        show (bool): Whether to display with matplotlib.
    """
    if hasattr(model, 'module'):
        model = model.module
    if _is_lane_result(result):
        img = _draw_lane_result(model, img, result, score_thr=score_thr)
    else:
        result = _to_cpu_result(result)
        img = model.show_result(img, result, score_thr=score_thr, show=False)
    if show:
        plt.figure(figsize=fig_size)
        plt.imshow(mmcv.bgr2rgb(img))
        plt.show()
    return img
