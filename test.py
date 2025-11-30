import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from _my_utils.nuScenes import nuScenesDataset_RadarDC_V6
from model_V6 import RadarDC
from _my_utils.metrics import my_metrics
from _my_utils.my_visualization_v3 import visualize_results_batch
import numpy as np
import time
import wandb
from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole


class Config:
    experiment_name = "Test_RadarDC_V6_BestModel"
    batch_size = 8
    CUDA_VISIBLE_DEVICES = '0'
    num_workers = 4
    checkpoint_path = f"./checkpoints_V6/RadarDC_V6_best.pth"  # Best model checkpoint
    test_image_paths = f"./data/testing/nuscenes/nuscenes_test_image.txt"
    test_radar_paths = f"./data/testing/nuscenes/nuscenes_test_radar.txt"
    test_ground_truth_paths = f"./data/testing/nuscenes/nuscenes_test_lidar.txt"
    test_K_cam_paths = f"./data/testing/nuscenes/nuscenes_test_k_cam.txt"
    
    # 시각화 관련 설정
    save_visualization = True  # 시각화 저장 여부
    visualization_dir = f"./test_results/{experiment_name}"  # 시각화 결과 저장 디렉토리
    vmax_depth = 80.0  # 최대 깊이 값
    cmap = 'inferno'  # 컬러맵
    save_every_n_batches = 1  # N 배치마다 시각화 저장 (1이면 모든 배치)


def setup_ddp():
    """DDP 초기화"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        is_distributed = True
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        is_distributed = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return is_distributed, rank, world_size, local_rank


def cleanup_ddp():
    """DDP 정리"""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_unidepth(device):
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitb14")
    model.eval()  # 항상 freeze
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    return model


def test_model(model, mde_model, test_dataloader, device, is_distributed, rank, world_size, config, total_samples):
    """Test 함수 (metrics 포함) + 시각화"""
    model.eval()
    test_loss = 0.0
    test_steps = 0

    # Metrics 집계
    total_mae, total_rmse, total_imae, total_irmse, total_delta1 = 0.0, 0.0, 0.0, 0.0, 0.0

    # Test 시작 로깅
    if not is_distributed or rank == 0:
        print(f"Test 시작 - 총 {len(test_dataloader)} 배치")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            # Forward pass
            image, radar, ground_truth, K_cam = batch
            image = image.to(device)
            radar = radar.to(device)
            ground_truth = ground_truth.to(device)
            K_cam = K_cam.to(device)
            camera = Pinhole(K=K_cam)

            with torch.no_grad():
                mde_output = mde_model.infer(image, camera, normalize=False)['depth']
                mde_features = mde_model.infer(image, camera, normalize=False)['depth_features'] # [8, 384, 42, 74]

            dense_gt, elevation_learned_radar, num_point_invariant = model(mde_output, mde_features, radar, image) # [B,1,H,W]
            
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

            test_steps += batch_size
            
            # 시각화 저장 (설정에 따라)
            if config.save_visualization:
                if (batch_idx + 1) % config.save_every_n_batches == 0:
                    # 배치 데이터를 numpy로 변환
                    img_batch = image.detach().cpu().numpy()  # [B, C, H, W]
                    lidar_GT_batch = ground_truth.detach().cpu().numpy()  # [B, 1, H, W]
                    pred_batch = dense_gt.detach().cpu().numpy()  # [B, 1, H, W]
                    radar_batch = radar.detach().cpu().numpy()  # [B, 1, H, W]
                    
                    # 시각화 저장
                    # DDP 환경에서 rank 정보를 포함한 start_idx 계산
                    if is_distributed:
                        # 각 rank가 처리하는 샘플 수 = total_samples / world_size
                        samples_per_rank = total_samples // world_size
                        start_idx = batch_idx * config.batch_size + rank * samples_per_rank
                    else:
                        start_idx = batch_idx * config.batch_size
                    
                    visualize_results_batch(
                        vmax_depth=config.vmax_depth,
                        cmap=config.cmap,
                        img_batch=img_batch,
                        lidar_GT_batch=lidar_GT_batch,
                        pred_batch=pred_batch,
                        start_idx=start_idx,
                        input_radar_batch=radar_batch,
                        dirpath=config.visualization_dir
                    )
            
            # Test 진행 상황 로깅 (매 10배치마다)
            if (not is_distributed or rank == 0) and (batch_idx + 1) % 1 == 0:
                progress = (batch_idx + 1) / len(test_dataloader) * 100
                print(f"Test 진행: {batch_idx + 1}/{len(test_dataloader)} ({progress:.1f}%)")

    # DDP 환경에서 metrics와 loss 평균 계산
    if is_distributed:
        metrics_to_reduce = torch.tensor(
            [test_loss, total_mae, total_rmse, total_imae, total_irmse, total_delta1, test_steps],
            device=device
        )
        dist.all_reduce(metrics_to_reduce, op=dist.ReduceOp.SUM)
        test_loss, total_mae, total_rmse, total_imae, total_irmse, total_delta1, test_steps = metrics_to_reduce.tolist()

    # 평균 계산
    avg_loss = test_loss / max(test_steps, 1)
    avg_mae = total_mae / max(test_steps, 1)
    avg_rmse = total_rmse / max(test_steps, 1)
    avg_imae = total_imae / max(test_steps, 1)
    avg_irmse = total_irmse / max(test_steps, 1)
    avg_delta1 = total_delta1 / max(test_steps, 1)

    # 메인 프로세스에서만 결과 출력
    if not is_distributed or rank == 0:
        print(f"Test 완료!")
        print(f"[Test] loss: {avg_loss:.4f}, mae: {avg_mae:.4f}, rmse: {avg_rmse:.4f}, imae: {avg_imae:.4f}, irmse: {avg_irmse:.4f}, delta1: {avg_delta1:.2f}")

    return avg_loss, avg_mae, avg_rmse, avg_imae, avg_irmse, avg_delta1


def main():
    # Wandb API Key 설정
    os.environ['WANDB_API_KEY'] = 'your_wandb_api_key_here'
    
    # Config 인스턴스 생성
    config = Config()
    
    # 환경변수 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES
    
    # DDP 초기화
    is_distributed, rank, world_size, local_rank = setup_ddp()
    
    # Wandb 초기화 (rank 0에서만)
    if not is_distributed or rank == 0:
        wandb.init(
            project="RadarDC",
            name=config.experiment_name,
            config={
                "test_batch_size": config.batch_size,
                "checkpoint_path": config.checkpoint_path,
                "test_image_paths": config.test_image_paths,
                "test_radar_paths": config.test_radar_paths,
                "test_ground_truth_paths": config.test_ground_truth_paths,
                "test_K_cam_paths": config.test_K_cam_paths,
                "results_dirpath": config.visualization_dir
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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    
    # Checkpoint 로드
    if not is_distributed or rank == 0:
        print(f"Loading checkpoint from {config.checkpoint_path}")
    
    if os.path.exists(config.checkpoint_path):
        checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
        
        # 모델 상태 로드
        if is_distributed:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        if not is_distributed or rank == 0:
            print(f"Checkpoint loaded successfully!")
            print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
            print(f"Validation RMSE: {checkpoint.get('val_rmse', 'Unknown')}")
    else:
        if not is_distributed or rank == 0:
            print(f"Checkpoint not found at {config.checkpoint_path}")
            return
    
    # Test 데이터셋 생성
    test_dataset = nuScenesDataset_RadarDC_V6(
        image_paths=config.test_image_paths,
        radar_paths=config.test_radar_paths,
        ground_truth_paths=config.test_ground_truth_paths,
        K_cam_paths=config.test_K_cam_paths,
        mode='test'  # Test mode (no augmentation)
    )
    
    # DistributedSampler 설정
    if is_distributed:
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size, 
            sampler=test_sampler,
            num_workers=config.num_workers,
            pin_memory=True
        )
    else:
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size, 
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
    
    # Test 시작
    start_time = time.time()
    
    if not is_distributed or rank == 0:
        print(f"\n=== Test 시작 ===")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Batch size: {config.batch_size}")
        print(f"Device: {device}")
        print("=" * 50)
    
    # Test 실행
    test_loss, test_mae, test_rmse, test_imae, test_irmse, test_delta1 = test_model(
        model, mde_model, test_dataloader, device, is_distributed, rank, world_size, config, len(test_dataset)
    )
    
    test_time = time.time() - start_time
    
    # 메인 프로세스에서만 최종 결과 출력
    if not is_distributed or rank == 0:
        print(f"\n=== Test 결과 ===")
        print(f"Test 시간: {test_time:.2f}초 ({test_time/60:.2f}분)")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test iMAE: {test_imae:.4f}")
        print(f"Test iRMSE: {test_irmse:.4f}")
        print(f"Test δ1: {test_delta1:.2f}%")
        print("=" * 50)
        
        # Wandb에 test 결과 로깅
        wandb.log({
            "test/loss": test_loss,
            "test/mae": test_mae,
            "test/rmse": test_rmse,
            "test/imae": test_imae,
            "test/irmse": test_irmse,
            "test/delta1": test_delta1,
            "test/time": test_time,
            "test/samples": len(test_dataset)
        })
        
        # 결과를 파일로 저장
        result_file = f"test_results_{config.experiment_name}.txt"
        with open(result_file, 'w') as f:
            f.write(f"Test Results for {config.experiment_name}\n")
            f.write(f"Checkpoint: {config.checkpoint_path}\n")
            f.write(f"Test samples: {len(test_dataset)}\n")
            f.write(f"Test time: {test_time:.2f} seconds\n")
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test MAE: {test_mae:.4f}\n")
            f.write(f"Test RMSE: {test_rmse:.4f}\n")
            f.write(f"Test iMAE: {test_imae:.4f}\n")
            f.write(f"Test iRMSE: {test_irmse:.4f}\n")
            f.write(f"Test δ1: {test_delta1:.2f}%\n")
        
        print(f"Results saved to {result_file}")
        
        # Wandb 종료
        wandb.finish()

    # DDP 정리
    cleanup_ddp()


if __name__ == "__main__":
    main()