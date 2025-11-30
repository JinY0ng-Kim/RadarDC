import torch
import torch.nn.functional as F


class DepthLoss:
    """Depth estimation을 위한 Loss 함수들"""
    
    @staticmethod
    def smooth_l1_loss(pred_depth, gt_depth, reduction='mean'):
        """Smooth L1 Loss (Huber Loss)"""
        return F.smooth_l1_loss(pred_depth, gt_depth, reduction=reduction)
    
    @staticmethod
    def l1_loss(pred_depth, gt_depth, reduction='mean'):
        """L1 Loss (Mean Absolute Error)"""
        return F.l1_loss(pred_depth, gt_depth, reduction=reduction)
    
    @staticmethod
    def mse_loss(pred_depth, gt_depth, reduction='mean'):
        """MSE Loss (Mean Squared Error)"""
        return F.mse_loss(pred_depth, gt_depth, reduction=reduction)
    
    @staticmethod
    def l2_loss(pred_depth, gt_depth, reduction='mean'):
        """L2 Loss (Mean Squared Error) - MSE와 동일"""
        return F.mse_loss(pred_depth, gt_depth, reduction=reduction)
    
    @staticmethod
    def compute_loss(pred_depth, gt_depth, loss_type='smooth_l1'):
        """Loss 계산 함수"""
        if loss_type == 'smooth_l1':
            return DepthLoss.smooth_l1_loss(pred_depth, gt_depth)
        elif loss_type == 'l1':
            return DepthLoss.l1_loss(pred_depth, gt_depth)
        elif loss_type == 'mse':
            return DepthLoss.mse_loss(pred_depth, gt_depth)
        elif loss_type == 'l2':
            return DepthLoss.l2_loss(pred_depth, gt_depth)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


class MultiTaskLoss:
    """다중 Loss 계산 클래스"""
    
    def __init__(self, loss_weights=None):
        """
        Args:
            loss_weights: 각 loss의 가중치 딕셔너리
                예: {'depth_l1': 1.0, 'depth_l2': 0.5}
        """
        self.loss_weights = loss_weights or {
            'depth_l1': 1.0,
            'depth_l2': 0.5
        }
    
    def compute_multiple_losses(self, pred_depth, gt_depth):
        """다중 Loss 계산 (valid 영역에서만)"""
        losses = {} 

        # Valid mask 생성 (gt_depth > 0인 영역)
        lidar_valid_mask = gt_depth > 0
        
        # Valid 영역이 있는지 확인
        if lidar_valid_mask.sum() == 0:
            # Valid 영역이 없는 경우 0 loss 반환
            losses['depth_l1'] = torch.tensor(0.0, device=pred_depth.device, requires_grad=True)
            losses['depth_l2'] = torch.tensor(0.0, device=pred_depth.device, requires_grad=True)
        else:
            # Valid 영역에서만 loss 계산
            losses['depth_l1'] = DepthLoss.l1_loss(pred_depth[lidar_valid_mask], gt_depth[lidar_valid_mask])
            losses['depth_l2'] = DepthLoss.l2_loss(pred_depth[lidar_valid_mask], gt_depth[lidar_valid_mask])
        
        return losses
    
    def compute_weighted_loss(self, pred_depth, gt_depth):
        """가중치가 적용된 총 Loss 계산"""
        losses = self.compute_multiple_losses(pred_depth, gt_depth)
        
        # 가중치 적용
        weighted_loss = 0.0
        for loss_name, loss_value in losses.items():
            if loss_name in self.loss_weights:
                weighted_loss += self.loss_weights[loss_name] * loss_value
        
        return weighted_loss, losses


# class DepthMetrics:
#     """Depth estimation 평가 메트릭들"""
    
#     @staticmethod
#     def abs_rel_error(pred_depth, gt_depth, mask=None):
#         """Absolute Relative Error"""
#         if mask is not None:
#             pred_depth = pred_depth[mask]
#             gt_depth = gt_depth[mask]
        
#         return torch.mean(torch.abs(pred_depth - gt_depth) / gt_depth)
    
#     @staticmethod
#     def sq_rel_error(pred_depth, gt_depth, mask=None):
#         """Squared Relative Error"""
#         if mask is not None:
#             pred_depth = pred_depth[mask]
#             gt_depth = gt_depth[mask]
        
#         return torch.mean(((pred_depth - gt_depth) ** 2) / gt_depth)
    
#     @staticmethod
#     def rmse(pred_depth, gt_depth, mask=None):
#         """Root Mean Square Error"""
#         if mask is not None:
#             pred_depth = pred_depth[mask]
#             gt_depth = gt_depth[mask]
        
#         return torch.sqrt(torch.mean((pred_depth - gt_depth) ** 2))
    
#     @staticmethod
#     def rmse_log(pred_depth, gt_depth, mask=None):
#         """Root Mean Square Error in log space"""
#         if mask is not None:
#             pred_depth = pred_depth[mask]
#             gt_depth = gt_depth[mask]
        
#         return torch.sqrt(torch.mean((torch.log(pred_depth) - torch.log(gt_depth)) ** 2))
    
#     @staticmethod
#     def compute_all_metrics(pred_depth, gt_depth, mask=None):
#         """모든 메트릭 계산"""
#         metrics = {}
#         metrics['abs_rel'] = DepthMetrics.abs_rel_error(pred_depth, gt_depth, mask)
#         metrics['sq_rel'] = DepthMetrics.sq_rel_error(pred_depth, gt_depth, mask)
#         metrics['rmse'] = DepthMetrics.rmse(pred_depth, gt_depth, mask)
#         metrics['rmse_log'] = DepthMetrics.rmse_log(pred_depth, gt_depth, mask)
        
#         return metrics
