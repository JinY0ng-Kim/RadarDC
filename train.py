import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from _my_utils.nuScenes import nuScenesDataset_RadarDC_V6
from model_V6 import RadarDC
from _my_utils.metrics import my_metrics
from loss import MultiTaskLoss
import wandb
from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

class Config:
    experiment_name = "RadarDC_V6"
    batch_size = 8
    CUDA_VISIBLE_DEVICES = '0,1,2,3,4,5,6,7,8'
    num_workers = 4
    OMP_NUM_THREADS = 4
    
    # Training parameters
    epochs = 100
    learning_rate = 7e-3  
    weight_decay = 1e-4  
    save_interval = 10 
    log_interval = 1  
    val_interval = 1  

def set_seed(seed=42):
    """재현성을 위한 seed 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_unidepth(device):
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitb14")
    model.eval()  # 항상 freeze
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    return model

def visualize_depth_maps(radar, elevation_learned_radar, final_depth, ground_truth, mde_output, image, max_depth=80, save_path=None):
    """
    Radar, elevation_learned_radar, final_depth, ground_truth, mde_output, image을 inferno colormap으로 시각화
    """
    # Convert to numpy and squeeze batch dimension
    radar_np = radar.squeeze(0).squeeze(0).detach().cpu().numpy()  # [H, W]
    elevation_np = elevation_learned_radar.squeeze(0).squeeze(0).detach().cpu().numpy()  # [H, W]
    final_np = final_depth.squeeze(0).squeeze(0).detach().cpu().numpy()  # [H, W]
    gt_np = ground_truth.squeeze(0).squeeze(0).detach().cpu().numpy()  # [H, W]
    mde_np = mde_output.squeeze(0).squeeze(0).detach().cpu().numpy()  # [H, W]
    image_np = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # [H, W, 3] for RGB
    
    # Clip to max_depth
    radar_np = np.clip(radar_np, 0, max_depth)
    elevation_np = np.clip(elevation_np, 0, max_depth)
    final_np = np.clip(final_np, 0, max_depth)
    gt_np = np.clip(gt_np, 0, max_depth)
    mde_np = np.clip(mde_np, 0, max_depth)
    
    # # Normalize image to [0, 1] for display
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    # 역정규화
    image_rgb_denorm = image_np * std + mean
    image_rgb_denorm = np.clip(image_rgb_denorm, 0, 1)
    # uint8 변환 후 저장
    image_np = image_rgb_denorm * 255.0
    image_np = image_np.astype(np.uint8)
    
    # Apply dilation for better visualization
    # Radar and elevation_learned_radar: 7x7 dilation
    radar_dilated = cv2.dilate(radar_np, np.ones((15, 15), np.uint8))
    elevation_dilated = cv2.dilate(elevation_np, np.ones((15, 15), np.uint8))
    
    # Ground truth: 7x7 dilation
    gt_dilated = cv2.dilate(gt_np, np.ones((7, 7), np.uint8))
    
    # MDE output: no dilation (keep original)
    mde_vis = mde_np
    
    # Final depth: no dilation (keep original)
    final_vis = final_np
    
    # Create figure with subplots (3x3 layout)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Depth Maps Visualization (Inferno Colormap)', fontsize=16)
    
    # Row 1: Input data
    # Input Image (original)
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    # Radar (7x7 dilated)
    im1 = axes[0, 1].imshow(radar_dilated, cmap='inferno', vmin=0, vmax=max_depth)
    axes[0, 1].set_title('Input Radar (15x15 dilated)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Elevation Learned Radar (7x7 dilated)
    im2 = axes[0, 2].imshow(elevation_dilated, cmap='inferno', vmin=0, vmax=max_depth)
    axes[0, 2].set_title('Elevation Learned Radar (15x15 dilated)')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Row 2: Model outputs
    # MDE Output (original)
    im3 = axes[1, 0].imshow(mde_vis, cmap='inferno', vmin=0, vmax=max_depth)
    axes[1, 0].set_title('MDE Output (original)')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Final Depth (no dilation)
    im4 = axes[1, 1].imshow(final_vis, cmap='inferno', vmin=0, vmax=max_depth)
    axes[1, 1].set_title('Final Depth (Model Output)')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Ground Truth (7x7 dilated)
    im5 = axes[1, 2].imshow(gt_dilated, cmap='inferno', vmin=0, vmax=max_depth)
    axes[1, 2].set_title('Ground Truth (7x7 dilated)')
    axes[1, 2].axis('off')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    # Row 3: Overlap visualizations
    # RGB + Elevation overlapped
    # Show RGB image first, then overlay elevation on top
    axes[2, 0].imshow(image_np)
    # Overlay elevation with transparency
    elevation_normalized = elevation_dilated / max_depth
    im6 = axes[2, 0].imshow(elevation_normalized, cmap='inferno', alpha=0.6, vmin=0, vmax=1)
    axes[2, 0].set_title('RGB + Elevation Overlapped')
    axes[2, 0].axis('off')
    
    # MDE + Elevation overlapped (overlay visualization)
    # Show MDE first, then overlay elevation on top
    axes[2, 1].imshow(mde_vis, cmap='inferno', vmin=0, vmax=max_depth)
    # Overlay elevation with transparency
    elevation_normalized_mde = elevation_dilated / max_depth
    im7 = axes[2, 1].imshow(elevation_normalized_mde, cmap='inferno', alpha=0.6, vmin=0, vmax=1)
    axes[2, 1].set_title('MDE + Elevation Overlapped')
    axes[2, 1].axis('off')
    plt.colorbar(im7, ax=axes[2, 1], fraction=0.046, pad=0.04)
    
    # Elevation + Ground Truth overlapped
    # Show elevation first
    axes[2, 2].imshow(elevation_dilated, cmap='inferno', vmin=0, vmax=max_depth)
    # Overlay ground truth with transparency
    gt_normalized = gt_dilated / max_depth
    im8 = axes[2, 2].imshow(gt_normalized, cmap='inferno', alpha=0.6, vmin=0, vmax=1)
    axes[2, 2].set_title('Elevation + GT Overlapped')
    axes[2, 2].axis('off')
    plt.colorbar(im8, ax=axes[2, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    # Print statistics (using dilated images for radar, elevation, and GT; original for mde and final)
    print(f"\nDepth Statistics:")
    print(f"Image - Min: {image_np.min():.3f}, Max: {image_np.max():.3f}, Mean: {image_np.mean():.3f}")
    print(f"Radar (7x7 dilated) - Min: {radar_dilated.min():.3f}, Max: {radar_dilated.max():.3f}, Mean: {radar_dilated.mean():.3f}")
    print(f"Elevation (7x7 dilated) - Min: {elevation_dilated.min():.3f}, Max: {elevation_dilated.max():.3f}, Mean: {elevation_dilated.mean():.3f}")
    print(f"MDE (original) - Min: {mde_vis.min():.3f}, Max: {mde_vis.max():.3f}, Mean: {mde_vis.mean():.3f}")
    print(f"Final (original) - Min: {final_vis.min():.3f}, Max: {final_vis.max():.3f}, Mean: {final_vis.mean():.3f}")
    print(f"GT (7x7 dilated) - Min: {gt_dilated.min():.3f}, Max: {gt_dilated.max():.3f}, Mean: {gt_dilated.mean():.3f}")
    print(f"RGB + Elevation Overlapped - Elevation normalized - Min: {elevation_normalized.min():.3f}, Max: {elevation_normalized.max():.3f}, Mean: {elevation_normalized.mean():.3f}")
    print(f"MDE + Elevation Overlapped - MDE - Min: {mde_vis.min():.3f}, Max: {mde_vis.max():.3f}, Mean: {mde_vis.mean():.3f}")
    print(f"MDE + Elevation Overlapped - Elevation normalized - Min: {elevation_normalized_mde.min():.3f}, Max: {elevation_normalized_mde.max():.3f}, Mean: {elevation_normalized_mde.mean():.3f}")
    print(f"Elevation + GT Overlapped - Elevation - Min: {elevation_dilated.min():.3f}, Max: {elevation_dilated.max():.3f}, Mean: {elevation_dilated.mean():.3f}")
    print(f"Elevation + GT Overlapped - GT normalized - Min: {gt_normalized.min():.3f}, Max: {gt_normalized.max():.3f}, Mean: {gt_normalized.mean():.3f}")


def setup_ddp():
    """DDP 초기화"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return True, rank, world_size, local_rank


def cleanup_ddp():
    """DDP 정리"""
    if dist.is_initialized():
        dist.destroy_process_group()


def validate_model(model, mde_model, val_dataloader, device, is_distributed, rank, world_size, multi_loss):
    """검증 함수 (metrics 포함)"""
    model.eval()
    val_loss = 0.0
    val_steps = 0

    # Metrics 집계
    total_mae, total_rmse, total_imae, total_irmse, total_delta1 = 0.0, 0.0, 0.0, 0.0, 0.0

    # Validation 시작 로깅
    if not is_distributed or rank == 0:
        print(f"Validation 시작 - 총 {len(val_dataloader)} 배치")

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            # 데이터를 디바이스로 이동 (tuple 형태로 반환됨)
            image, radar, ground_truth, K_cam = batch
            image = image.to(device)
            radar = radar.to(device)
            ground_truth = ground_truth.to(device)
            K_cam = K_cam.to(device)
            camera = Pinhole(K=K_cam)

            # Forward pass
            with torch.no_grad():
                mde_output = mde_model.infer(image, camera, normalize=False)['depth']
                mde_features = mde_model.infer(image, camera, normalize=False)['depth_features'] # [8, 384, 42, 74]

            dense_gt, elevation_learned_radar, num_point_invariant = model(mde_output, mde_features, radar, image) # [B,1,H,W]
            # Loss 계산 (training과 동일한 multi_loss 사용)
            weighted_loss, losses = multi_loss.compute_weighted_loss(
                dense_gt, ground_truth
            )
            loss = weighted_loss
            
            val_loss += loss.item()

            # Metrics 계산 (_my_utils.metrics.py 사용)
            # (배치 내 모든 sample에 대해 metrics)
            pred_np = dense_gt.squeeze(1).detach().cpu().numpy()  # [B,H,W]
            gt_np = ground_truth.squeeze(1).detach().cpu().numpy()  # [B,H,W]
            batch_size = pred_np.shape[0]
            for i in range(batch_size):
                mae, rmse, imae, irmse, delta1 = my_metrics(pred_np[i], gt_np[i], max_depth=80)
                total_mae += mae
                total_rmse += rmse
                total_imae += imae
                total_irmse += irmse
                total_delta1 += delta1

            val_steps += 1
            
            # Validation 로깅 (training과 동일한 방식)
            if (not is_distributed or rank == 0) and batch_idx % Config.log_interval == 0:
                avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0
                
                # 개별 loss 값들 출력
                loss_breakdown = ""
                for loss_name, loss_value in losses.items():
                    loss_breakdown += f", {loss_name}: {loss_value.item():.6f}"
                
                print(f"Validation - batch {batch_idx+1}/{len(val_dataloader)}, "
                      f"Total Loss: {loss.item():.6f} Avg Loss: {avg_val_loss:.6f}{loss_breakdown}")

    # Training과 동일한 방식으로 loss 계산 (DDP 합산 없이)
    # val_steps는 배치 수이므로, 실제 샘플 수는 val_steps * batch_size
    total_samples = val_steps * Config.batch_size if val_steps > 0 else 1
    avg_loss = val_loss / val_steps if val_steps > 0 else 0.0
    avg_mae = total_mae / total_samples if total_samples > 0 else 0.0
    avg_rmse = total_rmse / total_samples if total_samples > 0 else 0.0
    avg_imae = total_imae / total_samples if total_samples > 0 else 0.0
    avg_irmse = total_irmse / total_samples if total_samples > 0 else 0.0
    avg_delta1 = total_delta1 / total_samples if total_samples > 0 else 0.0

    # Validation 완료 로깅
    if not is_distributed or rank == 0:
        print(f"Validation 완료!")
        print(f"[Validation] Total Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}, IMAE: {avg_imae:.4f}, IRMSE: {avg_irmse:.4f}, δ1: {avg_delta1:.2f}")

    return avg_loss, avg_mae, avg_rmse, avg_imae, avg_irmse, avg_delta1

def main():
    # Seed 고정 (재현성 보장)
    set_seed(42)
    
    # Wandb API Key 설정
    os.environ['WANDB_API_KEY'] = 'your_wandb_api_key_here' 
    
    # 환경변수 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = Config.CUDA_VISIBLE_DEVICES
    os.environ['OMP_NUM_THREADS'] = str(Config.OMP_NUM_THREADS)
    
    # DDP 초기화
    is_distributed, rank, world_size, local_rank = setup_ddp()
    
    # Wandb 초기화 (rank 0에서만)
    if not is_distributed or rank == 0:
        wandb.init(
            project="RadarDC",
            name=Config.experiment_name,
            config={
                "learning_rate": Config.learning_rate,
                "batch_size": Config.batch_size,
                "epochs": Config.epochs,
                "weight_decay": Config.weight_decay,
                "val_interval": Config.val_interval
            }
        )
    
    # RadarDC 모델 로드
    model = RadarDC()

    # GPU 설정
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        model = model.to(device)
        mde_model = load_unidepth(device)
    else:
        device = torch.device('cpu')
        model = model.to(device)
        mde_model = load_unidepth(device)
    
    # DDP로 모델 래핑
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    
    if not is_distributed or rank == 0:
        # 모델 summary 출력
        print("\n" + "="*50)
        print("MODEL SUMMARY")
        print("="*50)
        
        # 전체 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")
        print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
        
        # 모델 구조 출력
        print(f"\nModel architecture:")
        print(model)
        print("="*50 + "\n")
    
    # 옵티마이저 및 스케줄러 설정 (더 안정적인 설정)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=Config.learning_rate, 
        weight_decay=Config.weight_decay,
        betas=(0.9, 0.999),  # 더 안정적인 beta 값
        eps=1e-8
    )
    
    # 더 적극적인 learning rate 감소
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    
    # Multi-task loss 설정 (dmde loss 제거)
    multi_loss = MultiTaskLoss(loss_weights={
        'depth_l1': 1.0,
        'depth_l2': 0.4,
    })
    val_multi_loss = MultiTaskLoss(loss_weights={
        'depth_l1': 1.0,
        'depth_l2': 0.4,
    })

    # 데이터셋 생성
    train_dataset = nuScenesDataset_RadarDC_V6(
        image_paths="./data/training/nuscenes/nuscenes_train_image.txt",
        radar_paths="./data/training/nuscenes/nuscenes_train_radar.txt",
        ground_truth_paths="./data/training/nuscenes/nuscenes_train_lidar.txt",
        K_cam_paths="./data/training/nuscenes/nuscenes_train_k_cam.txt",
        mode='train'
    )

    val_dataset = nuScenesDataset_RadarDC_V6(
        image_paths="./data/testing/nuscenes/nuscenes_test_image.txt",
        radar_paths="./data/testing/nuscenes/nuscenes_test_radar.txt",
        ground_truth_paths="./data/testing/nuscenes/nuscenes_test_lidar.txt",
        K_cam_paths="./data/testing/nuscenes/nuscenes_test_k_cam.txt",
        mode='val'
    )

    # DistributedSampler 설정
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=Config.batch_size, 
            sampler=train_sampler,
            num_workers=Config.num_workers,
            pin_memory=True
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=Config.batch_size, 
            sampler=val_sampler,
            num_workers=Config.num_workers,
            pin_memory=True
        )
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
    
    # 메인 프로세스에서만 출력
    if not is_distributed or rank == 0:
        print(f"Training with {len(train_dataset)} samples")
        print(f"Validation with {len(val_dataset)} samples")
        print(f"Training per GPU: {len(train_dataloader)}")
        print(f"Validation per GPU: {len(val_dataloader)}")
        print(f"Using device: {device}")
        if is_distributed:
            print(f"Distributed training with {world_size} GPUs")
    
    # 훈련 시작
    start_time = time.time()
    best_rmse = float('inf')  # 최고 성능 추적
    
    for epoch in range(Config.epochs):
        epoch_start_time = time.time()
        
        # 훈련 모드
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        # DistributedSampler 설정 (매 epoch마다)
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        for step, batch in enumerate(train_dataloader):
            step_start_time = time.time()
            
            # 데이터를 디바이스로 이동 (tuple 형태로 반환됨)
            image, radar, ground_truth, K_cam = batch
            image = image.to(device)
            radar = radar.to(device)
            ground_truth = ground_truth.to(device)
            K_cam = K_cam.to(device)

            camera = Pinhole(K=K_cam)
            

            # Forward pass
            optimizer.zero_grad()
            
            with torch.no_grad():
                mde_output = mde_model.infer(image, camera, normalize=False)['depth']
                mde_features = mde_model.infer(image, camera, normalize=False)['depth_features'] # [8, 384, 42, 74]

            # 모델 forward pass
            dense_gt, elevation_learned_radar, num_point_invariant = model(mde_output, mde_features, radar, image)
            
            # (디버그용)
            if (step == 0 or step % 50 == 0) and (not is_distributed or rank == 0):
                print("\n" + "="*50)
                print("DEBUG VISUALIZATION")
                print("="*50)
                save_path = f"debug_visualization_V6/debug_visualization_epoch_{epoch+1}_step_{step+1}.png"
                visualize_depth_maps(radar[0], elevation_learned_radar[0], dense_gt[0], ground_truth[0], mde_output[0], image[0],
                                   max_depth=80, save_path=save_path)
                print("="*50 + "\n")
            
            # Multi-task loss 계산 (dense_gt와 dmde_output 모두 사용)
            weighted_loss, losses = multi_loss.compute_weighted_loss(
                dense_gt, ground_truth
            )
            loss = weighted_loss
            
            # Backward pass
            loss.backward()

            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            step_time = time.time() - step_start_time
            

            avg_loss = train_loss / train_steps if train_steps > 0 else 0.0

            # 로깅
            if (not is_distributed or rank == 0) and step % Config.log_interval == 0:
                avg_train_loss = avg_loss
                
                # 개별 loss 값들 출력
                loss_breakdown = ""
                for loss_name, loss_value in losses.items():
                    loss_breakdown += f", {loss_name}: {loss_value.item():.6f}"
                
                print(f"Epoch {epoch+1}/{Config.epochs}, step {step+1}/{len(train_dataloader)}, "
                      f"Total Loss: {loss.item():.6f} Avg Loss: {avg_loss:.6f}{loss_breakdown}, Step Time: {step_time:.3f}s, "
                      f"Point Invariant: {num_point_invariant}")
                
                # Wandb 로깅 (개별 loss 컴포넌트 포함)
                wandb_log = {
                    "train/total_loss": loss.item()/Config.batch_size,
                    "train/avg_loss": avg_train_loss,
                    "train/step_time": step_time,
                    "epoch": epoch + 1,
                    "step": step + 1
                }
                
                # 개별 loss 컴포넌트 추가
                for loss_name, loss_value in losses.items():
                    wandb_log[f"train/{loss_name}"] = loss_value.item()/Config.batch_size
                
                wandb.log(wandb_log)
        
        # 검증 (val_interval마다)
        if (epoch + 1) % Config.val_interval == 0:
            val_loss, val_mae, val_rmse, val_imae, val_irmse, val_delta1 = validate_model(model, mde_model, val_dataloader, device, is_distributed, rank, world_size, val_multi_loss)
            
            # Wandb validation 로깅
            if not is_distributed or rank == 0:
                wandb.log({
                    "val/loss": val_loss,
                    "val/mae": val_mae,
                    "val/rmse": val_rmse,
                    "val/imae": val_imae,
                    "val/irmse": val_irmse,
                    "val/delta1": val_delta1,
                    "epoch": epoch + 1
                })
            
            # 최고 성능 모델 저장 (val_rmse 기준)
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                if not is_distributed or rank == 0:
                    best_save_path = f"checkpoints_V6/{Config.experiment_name}_best.pth"
                    os.makedirs("checkpoints_V6", exist_ok=True)
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.module.state_dict() if is_distributed else model.state_dict(),
                        'val_rmse': val_rmse,
                        'val_mae': val_mae,
                        'val_imae': val_imae,
                        'val_irmse': val_irmse,
                        'val_delta1': val_delta1,
                    }, best_save_path)
                    print(f"Best model saved to {best_save_path} (RMSE: {val_rmse:.4f})")
        else:
            # validation을 수행하지 않는 epoch에서는 기본값 설정
            val_loss, val_mae, val_rmse, val_imae, val_irmse, val_delta1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        # 스케줄러 업데이트 (validation loss 전달)
        if val_loss > 0:
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = train_loss / train_steps if train_steps > 0 else 0.0
        
        # 메인 프로세스에서만 출력
        if not is_distributed or rank == 0:
            print(f"\nEpoch {epoch+1}/{Config.epochs} 완료:")
            print(f"  Train Total Loss: {avg_train_loss:.6f}")
            if (epoch + 1) % Config.val_interval == 0:
                print(f"  Val Total Loss: {val_loss:.6f}")
                print(f"  Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}")
                print(f"  Val IMAE: {val_imae:.4f}, Val IRMSE: {val_irmse:.4f}, Val δ1: {val_delta1:.2f}")
                print(f"  Best RMSE: {best_rmse:.4f}")
            print(f"  Epoch Time: {epoch_time:.2f}s")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
            print("-" * 50)
        
        # 모델 저장
        if (not is_distributed or rank == 0) and (epoch + 1) % Config.save_interval == 0:
            save_path = f"checkpoints_V6/{Config.experiment_name}_epoch_{epoch+1}.pth"
            os.makedirs("checkpoints_V6", exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if is_distributed else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
            }, save_path)
            print(f"Model saved to {save_path}")
    
    total_time = time.time() - start_time
    
    # 메인 프로세스에서만 최종 출력
    if not is_distributed or rank == 0:
        print(f"\n훈련 완료!")
        print(f"총 훈련 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")
        print(f"평균 epoch 시간: {total_time/Config.epochs:.2f}초")
        
        # Wandb 종료
        wandb.finish()

    # DDP 정리
    cleanup_ddp()


if __name__ == "__main__":
    main()
