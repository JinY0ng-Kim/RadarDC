import numpy as np
from . import eval_utils
import torch

def my_metrics(output_depth, ground_truth, max_depth):
    if isinstance(output_depth, torch.Tensor):
        output_depth = output_depth.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    
    min_evaluate_depth = 0
    max_evaluate_depth = max_depth

    validity_map = np.where(ground_truth > 0, 1, 0)
    validity_mask = np.where(validity_map > 0, 1, 0)

    min_max_mask = np.logical_and(
        ground_truth > min_evaluate_depth,
        ground_truth < max_evaluate_depth)
    mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)
    
    mae = eval_utils.mean_abs_err(1000.0 * output_depth[mask], 1000.0 * ground_truth[mask])
    rmse = eval_utils.root_mean_sq_err(1000.0 * output_depth[mask], 1000.0 * ground_truth[mask])
    imae = eval_utils.inv_mean_abs_err(0.001 * output_depth[mask], 0.001 * ground_truth[mask])
    irmse = eval_utils.inv_root_mean_sq_err(0.001 * output_depth[mask], 0.001 * ground_truth[mask])

    # Calculate delta1 metric (percentage of pixels with max relative error < 1.25)
    pred_masked = output_depth[mask]
    gt_masked = ground_truth[mask]
    
    # Avoid division by zero
    gt_nonzero = gt_masked > 0
    if np.sum(gt_nonzero) > 0:
        pred_nonzero = pred_masked[gt_nonzero]
        gt_nonzero_vals = gt_masked[gt_nonzero]
        
        # Calculate relative error
        rel_error = np.maximum(pred_nonzero / gt_nonzero_vals, gt_nonzero_vals / pred_nonzero)
        delta1 = np.mean(rel_error < 1.25) * 100.0
    else:
        delta1 = 0.0

    return mae, rmse, imae, irmse, delta1


def my_metrics_uni(output_depth, ground_truth, max_depth):
    if isinstance(output_depth, torch.Tensor):
        output_depth = output_depth.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    
    min_evaluate_depth = 0
    max_evaluate_depth = max_depth

    # 간소화된 validity_mask 계산
    validity_mask = ground_truth > 0

    min_max_mask = np.logical_and(
        ground_truth > min_evaluate_depth,
        ground_truth < max_evaluate_depth)
    mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)
    
    mae = eval_utils.mean_abs_err(1000.0 * output_depth[mask], 1000.0 * ground_truth[mask])
    rmse = eval_utils.root_mean_sq_err(1000.0 * output_depth[mask], 1000.0 * ground_truth[mask])
    imae = eval_utils.inv_mean_abs_err(0.001 * output_depth[mask], 0.001 * ground_truth[mask])
    irmse = eval_utils.inv_root_mean_sq_err(0.001 * output_depth[mask], 0.001 * ground_truth[mask])


    pred_nonzero = output_depth[validity_mask]
    gt_nonzero_vals = ground_truth[validity_mask]
    
    # Calculate relative error
    rel_error = np.maximum(pred_nonzero / gt_nonzero_vals, gt_nonzero_vals / pred_nonzero)
    delta1 = np.mean(rel_error < 1.25) * 100.0


    # delta1_uni 계산 - PyTorch 텐서를 NumPy로 변환
    delta1_uni = delta(ground_truth[mask], output_depth[mask], exponent=1.0).item() * 100.0
    

    return mae, rmse, imae, irmse, delta1, delta1_uni

def delta(tensor1, tensor2, exponent):
    # numpy 배열을 PyTorch 텐서로 변환
    if isinstance(tensor1, np.ndarray):
        tensor1 = torch.from_numpy(tensor1).float()
    if isinstance(tensor2, np.ndarray):
        tensor2 = torch.from_numpy(tensor2).float()
     
    inlier = torch.maximum((tensor1 / tensor2), (tensor2 / tensor1))
    return (inlier < 1.25**exponent).to(torch.float32).mean()