from collections import defaultdict
import os.path as osp
import time
import pickle
import shutil
import tempfile
import numpy as np
import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results
from acdet.utils.structure_utils import weighted_loss

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    active_cycle=-1,
                    **kwargs):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(
                return_loss=False, 
                rescale=True, 
                active_cycle=active_cycle, 
                **data, 
                **{'show_kwargs': kwargs})
        results.extend(result)
        prog_bar.update()
    return results


def multi_gpu_test(model, 
                   data_loader, 
                   tmpdir=None, 
                   gpu_collect=False,
                   active_cycle=-1,
                   **kwargs):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(
                return_loss=False, 
                rescale=True, 
                active_cycle=active_cycle, 
                **data, 
                **{'show_kwargs': kwargs})
        results.extend(result)

        if rank == 0:
            for _ in range(world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def single_gpu_test_mcdropout_mean_entropy(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    active_cycle=-1,
                    **kwargs):
    model.eval()
    from torch.nn import Dropout
    for _, module in model.named_modules():
        if isinstance(module, Dropout):
            module.train()

    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(
                return_loss=False, 
                rescale=True, 
                **data)
        results.extend(result)

        prog_bar.update()
    return results


def multi_gpu_test_mcdropout_mean_entropy(model, 
                   data_loader, 
                   tmpdir=None, 
                   gpu_collect=False,
                   active_cycle=-1,
                   **kwargs):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    from torch.nn import Dropout
    for _, module in model.named_modules():
        if isinstance(module, Dropout):
            module.train()

    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(
                return_loss=False, 
                rescale=True, 
                **data)
        results.extend(result)

        if rank == 0:
            for _ in range(world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def single_gpu_test_ll(model,
                       data_loader,
                       show=False,
                       out_dir=None,
                       show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            _ = model(return_loss=False, rescale=True, **data)
            result = model.roi_head.bbox_head.get_learning_loss()
            result = result.view(result.size(0))
        batch_size = len(result)
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results

def multi_gpu_test_ll(model,
                      data_loader, 
                      tmpdir=None, 
                      gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            _ = model(return_loss=False, rescale=True, **data)
            result = model.module.roi_head.bbox_head.get_learning_loss()
            result = result.view(result.size(0)).cpu().numpy().tolist()
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def single_gpu_test_almdn(model,
                       data_loader,
                       show=False,
                       out_dir=None,
                       show_score_thr=0.3):
    model.eval()
    setattr(model.roi_head.bbox_head, "active", True)
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        batch_size = len(result)
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results

def multi_gpu_test_almdn(model,
                      data_loader, 
                      tmpdir=None, 
                      gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    setattr(model.module.roi_head.bbox_head, "active", True)
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def single_gpu_test_miaod(model,
                          data_loader,
                          show=False,
                          out_dir=None,
                          show_score_thr=0.3,
                          active_cycle=-1):
    import torch.nn as nn
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            cls_score, _ = model(
                return_loss=False, 
                rescale=True, 
                active_cycle=active_cycle, 
                **data)
            cls_score_1, cls_score_2, _ = cls_score
            y_head_f_1 = nn.Sigmoid()(cls_score_1)
            y_head_f_2 = nn.Sigmoid()(cls_score_2)
            loss_l2_p = (y_head_f_1 - y_head_f_2).pow(2)
            result = [loss_l2_p.mean().cpu().numpy()]
            # uncertainty = loss_l2_p.mean(dim=1)
            # arg = uncertainty.argsort()
            # result = uncertainty[arg[-1000:]].mean()
        batch_size = len(result)
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results

def multi_gpu_test_miaod(model,
                      data_loader, 
                      tmpdir=None, 
                      gpu_collect=False,
                      active_cycle=-1):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    import torch.nn as nn
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            cls_score, _ = model(
                return_loss=False, 
                rescale=True, 
                active_cycle=active_cycle, 
                **data)
            cls_score_1, cls_score_2, _ = cls_score
            y_head_f_1 = nn.Sigmoid()(cls_score_1)
            y_head_f_2 = nn.Sigmoid()(cls_score_2)
            loss_l2_p = (y_head_f_1 - y_head_f_2).pow(2)
            result = [loss_l2_p.mean().cpu().numpy()]
            # uncertainty = loss_l2_p.mean(dim=1)
            # arg = uncertainty.argsort()
            # result = uncertainty[arg[-1000:]].mean()
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def single_gpu_test_wbqbc(model,
                         data_loader,
                         show=False,
                         out_dir=None,
                         show_score_thr=0.3,
                         active_cycle=-1):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(
                return_loss=False, 
                rescale=True, 
                active_cycle=active_cycle, 
                **data)
            results_classes = list(map(list, zip(*list(map(list, zip(*result)))[0])))
            num_classes = len(results_classes)
            confs, margins = np.zeros(num_classes), np.zeros(num_classes)
            for class_i, class_results in enumerate(results_classes):
                total_boxes = np.concatenate(class_results, 0)
                if total_boxes.shape[0] == 0: continue # no det results for this class
                cls_box_confs = total_boxes[:, -1]
                cls_box_margins = []
                for member_i, det_boxes in enumerate(class_results):
                    if det_boxes.shape[0] == 0: continue
                    for det_box in det_boxes:
                        score = det_box[-1]
                        if score <= 0.1 or score >= 0.9: continue
                        aux_boxes = np.vstack(class_results[:member_i] + class_results[member_i + 1:])  # xyxy format
                        overlaps = bbox_overlaps(aux_boxes[:, :4], det_box[None, :4])
                        aux_boxes = aux_boxes[overlaps.squeeze(1) > 0.3]
                        if aux_boxes.shape[0] == 0: 
                            secondmax_score = 0
                        else:
                            secondmax_score = np.max(aux_boxes[:, -1])
                        cls_box_margins.append(np.abs(score - secondmax_score))
                margins[class_i] = sum(cls_box_margins)/total_boxes.shape[0]
                confs[class_i] = np.max(cls_box_confs)
            uncertainty = np.sum(confs*margins/np.sum(confs))
        results.append(uncertainty)

        assert len(result[0]) == 1
        prog_bar.update()
    return results

def multi_gpu_test_wbqbc(model,
                        data_loader, 
                        tmpdir=None, 
                        gpu_collect=False,
                        active_cycle=-1):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(
                return_loss=False, 
                rescale=True, 
                active_cycle=active_cycle, 
                **data)
            results_classes = list(map(list, zip(*list(map(list, zip(*result)))[0])))
            num_classes = len(results_classes)
            confs, margins = np.zeros(num_classes), np.zeros(num_classes)
            for class_i, class_results in enumerate(results_classes):
                total_boxes = np.concatenate(class_results, 0)
                if total_boxes.shape[0] == 0: continue # no det results for this class
                cls_box_confs = total_boxes[:, -1]
                cls_box_margins = []
                for member_i, det_boxes in enumerate(class_results):
                    if det_boxes.shape[0] == 0: continue
                    for det_box in det_boxes:
                        score = det_box[-1]
                        if score <= 0.1 or score >= 0.9: continue
                        aux_boxes = np.vstack(class_results[:member_i] + class_results[member_i + 1:])  # xyxy format
                        overlaps = bbox_overlaps(aux_boxes[:, :4], det_box[None, :4])
                        aux_boxes = aux_boxes[overlaps.squeeze(1) > 0.3]
                        if aux_boxes.shape[0] == 0: 
                            secondmax_score = 0
                        else:
                            secondmax_score = np.max(aux_boxes[:, -1])
                        cls_box_margins.append(np.abs(score - secondmax_score))
                margins[class_i] = sum(cls_box_margins)/total_boxes.shape[0]
                confs[class_i] = np.max(cls_box_confs)
            uncertainty = np.sum(confs*margins/np.sum(confs))
        results.append(uncertainty)
        if rank == 0:
            assert len(result[0]) == 1
            batch_size = 1
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results

def single_gpu_test_coreset(model,
                         data_loaders,
                         show=False,
                         out_dir=None,
                         show_score_thr=0.3,
                         metric=2, # default is l2
                         active_cycle=-1):
    model.eval()
    results = defaultdict(list)
    assert isinstance(data_loaders, dict) and 'labeled' in data_loaders and 'unlabeled' in data_loaders, "For coreset, we need 2 data loaders, 'labeled' and 'unlabeled' comprised in dict, respectively"
    # loader names are 'labeled' and 'unlabeled' by default
    for loader_name, data_loader in data_loaders.items():
        dataset = data_loader.dataset
        print(f"Active test {loader_name}")
        prog_bar = mmcv.ProgressBar(len(dataset))
        time.sleep(2)  # This line can prevent deadlock problem in some cases.
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                result = model(
                    return_loss=False, 
                    rescale=True, 
                    active_cycle=active_cycle, 
                    **data)
                batch_size = result.shape[0]
                results[loader_name].append(result)

            for _ in range(batch_size):
                    prog_bar.update()
    for loader_name, tensor_list in results.items():
        results[loader_name] = torch.cat(tensor_list, 0)
    indices = coreset_kcenter_greedy(results['labeled'], results['unlabeled'], float('inf'), p_norm=metric) # 'budget_num = inf' means giving each unlabeled sample a rank
    return indices

def multi_gpu_test_coreset(model,
                        data_loaders, 
                        tmpdir=None, 
                        gpu_collect=False,
                        metric=2, # default is l2
                        active_cycle=-1):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = defaultdict(list)
    assert isinstance(data_loaders, dict) and 'labeled' in data_loaders and 'unlabeled' in data_loaders, "For coreset, we need 2 data loaders, 'labeled' and 'unlabeled' comprised in dict, respectively"
    # loader names are 'labeled' and 'unlabeled' by default
    for loader_name, data_loader in data_loaders.items():
        dataset = data_loader.dataset
        rank, world_size = get_dist_info()
        if rank == 0:
            print(f"Active test {loader_name}")
            prog_bar = mmcv.ProgressBar(len(dataset))
        time.sleep(2)  # This line can prevent deadlock problem in some cases.
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                result = model(
                    return_loss=False, 
                    rescale=True, 
                    active_cycle=active_cycle, 
                    **data)
                batch_size = result.shape[0]
                results[loader_name].append(result)

            if rank == 0:
                for _ in range(batch_size*world_size):
                    prog_bar.update()

    # collect results from all ranks
    for loader_name, tensor_list in results.items():
        results[loader_name] = torch.cat(tensor_list, 0)

    if gpu_collect:
        for loader_name in results:
            results[loader_name] = collect_results_gpu(results[loader_name], len(data_loaders[loader_name].dataset))
    else:
        for loader_name in results:
            results[loader_name] = collect_results_cpu(results[loader_name], len(data_loaders[loader_name].dataset))
    if rank == 0:
        for loader_name in results:
            results[loader_name] = list(map(lambda x: x.cuda(rank).unsqueeze(0), results[loader_name]))

        indices, _ = coreset_kcenter_greedy(torch.cat(results['labeled'], 0), torch.cat(results['unlabeled'], 0), float('inf'), p_norm=metric) # 'budget_num = inf' means giving each unlabeled sample a rank
    dist.barrier()

    if rank != 0:
        return None
    else:
        return indices

def bbox_overlaps(bboxes1, bboxes2, eps=1e-6):
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.shape[-1] == 4 or bboxes1.shape[0] == 0)
    assert (bboxes2.shape[-1] == 4 or bboxes2.shape[0] == 0)

    rows = bboxes1.shape[-2]
    cols = bboxes2.shape[-2]

    if rows * cols == 0:
        return np.zeros((rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    lt = np.maximum(bboxes1[..., :, None, :2],
                    bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
    rb = np.minimum(bboxes1[..., :, None, 2:],
                    bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

    wh = np.clip(rb - lt, 0, None)
    overlap = wh[..., 0] * wh[..., 1]

    union = area1[..., None] + area2[..., None, :] - overlap

    union = np.maximum(union, eps)
    ious = overlap / union
    return ious

def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results

def torch_cosine_distance(feat_a, feat_b):
    inner_dot = torch.mm(feat_a, feat_b.t())
    norm_matrix = torch.mm(feat_a.norm(p=2,dim=-1, keepdim=True), feat_b.norm(p=2,dim=-1, keepdim=True).t())
    return 1 - inner_dot/norm_matrix

def distance_function(feat_a, feat_b, p):
    if(p == 'cosine'):
        return torch_cosine_distance(feat_a, feat_b)
    return torch.cdist(feat_a, feat_b, int(p))


def coreset_kcenter_greedy(labeled_feat, unlabeled_feat, budget_num, p_norm=2):
    print(f"Start computing coreset, metric is {p_norm}")
    unlabeled_num = unlabeled_feat.shape[0]
    max_indices = min(budget_num, unlabeled_num)
    labeled_feat = labeled_feat
    unlabeled_feat = unlabeled_feat
    distance = distance_function(labeled_feat, unlabeled_feat, p=p_norm)
    print(distance.shape)
    min_distance = torch.amin(distance, dim=0, keepdim=True)
    print(min_distance.shape)
    indices = []
    # prog_bar = mmcv.ProgressBar(budget_num if budget_num != float('inf') else unlabeled_num)
    while len(indices) < max_indices:
        index = torch.argmax(min_distance, dim=-1)
        min_distance[0, index] = float('-inf')
        indices.append(index)
        if len(indices) in [max_indices * (i+1)//10 for i in range(10)]:
            print(f"Computed {len(indices)}/{max_indices} samples.")
        # prog_bar.update()

        # update min_distance
        min_distance = torch.amin(torch.cat([min_distance, distance_function(unlabeled_feat[index:index+1],unlabeled_feat, p=p_norm)],dim=0), dim=0,keepdim=True)
    opt = min_distance.max()
    return indices, opt
