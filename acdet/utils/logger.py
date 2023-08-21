import logging
import os
import sys
from collections import Counter, Sequence
from typing import Tuple

import mmcv
import numpy as np
import torch
from mmcv.runner.dist_utils import get_dist_info
from mmcv.utils import get_logger
from mmdet.core.visualization import imshow_det_bboxes
from acdet.models.utils.bbox_utils import Transform2D

_log_counter = Counter()


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name="acdet", log_file=log_file, log_level=log_level)
    logger.propagate = False
    return logger


def _find_caller():
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join("utils", "logger.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = r"acdet"
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back


def color_transform(img_tensor, mean, std, to_rgb=False):
    img_np = img_tensor.detach().cpu().numpy().transpose((1, 2, 0)).astype(np.float32)
    return mmcv.imdenormalize(img_np, mean, std, to_bgr=to_rgb)


def log_image_with_boxes(
    tag: str,
    image: torch.Tensor,
    bboxes: torch.Tensor,
    bbox_tag: str = None,
    labels: torch.Tensor = None,
    losses: torch.Tensor = None,
    scores: dict = None,
    class_names: Tuple[str] = None,
    filename: str = None,
    img_norm_cfg: dict = None,
    teacher_metas: dict = None,
    backend: str = "auto",
    interval: int = 50,
    show_kwargs: dict = None,
    color: str = None
):
    rank, _ = get_dist_info()
    if rank != 0:
        return
    if show_kwargs is not None:
        backend = show_kwargs['backend']
    if backend is None:
        return
    _, key = _find_caller()
    _log_counter[key] += 1
    if not (interval == 1 or _log_counter[key] % interval == 1):
        return
    if backend == "auto":
        backend = "file"

    if backend == "file":
        root_dir = os.environ.get("WORK_DIR", ".")
        color = color if color is not None else 'green'
        filename, ext = filename.split('.')
        total_scores = f'{scores:.05f}'
        filename = filename + '_' + total_scores + '.' + ext
        imshow_det_bboxes(
            image,
            bboxes.cpu().detach().numpy(),
            labels.cpu().detach().numpy(),
            class_names=class_names,
            show=False,
            bbox_color=color,
            out_file=os.path.join(root_dir, bbox_tag, filename),
        )
    else:
        raise TypeError("backend must be file")


def log_every_n(msg: str, n: int = 50, level: int = logging.DEBUG, backend="auto"):
    """
    Args:
        msg (Any):
        n (int):
        level (int):
        name (str):
    """
    caller_module, key = _find_caller()
    _log_counter[key] += 1
    if n == 1 or _log_counter[key] % n == 1:
        get_root_logger().log(level, msg)
