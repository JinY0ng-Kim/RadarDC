import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import matplotlib.colors as mcolors

def save_image_exact(data, save_path, dpi=100, cmap='inferno', vmin=0.0, vmax=80.0):
    """
    이미지 데이터를 여백 없이 원본 해상도 그대로 저장하는 함수
    
    Args:
        data (np.array): 저장할 이미지 데이터 (H, W) 또는 (H, W, C)
        save_path (str): 저장할 파일 경로
        dpi (int): 저장할 이미지의 DPI
        cmap (str, optional): 2D 데이터에 사용할 컬러맵
        vmin (float, optional): 컬러맵의 최소값
        vmax (float, optional): 컬러맵의 최대값
    """
    if data.ndim == 3: # 컬러 이미지 (H, W, C)
        H, W, C = data.shape
    else: # 흑백 또는 depth 이미지 (H, W)
        H, W = data.shape

    fig_size_in_inches = (W / dpi, H / dpi)
    
    fig, ax = plt.subplots(figsize=fig_size_in_inches, dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# def visualize_results(vmax_depth, cmap, img, lidar_GT, pred, ind, input_radar, dirpath):
#     dilation_gt_kernel = np.ones((7, 7), np.uint8)
#     dilation_kernel = np.ones((5, 5), np.uint8)

#     color_map = plt.get_cmap(cmap)
#     # colors = color_map(np.linspace(0, 1, 256))
#     # colors[0] = np.array([0, 0, 0, 1])  # 첫 번째 색을 검정으로 (RGBA)
#     # color_map = mcolors.ListedColormap(colors)
    
#     # --- Visualization ---
#     results_dirpath = dirpath

#     image_save_dir = os.path.join(results_dirpath, 'image')
#     depth_save_dir = os.path.join(results_dirpath, 'output')
#     ground_truth_save_dir = os.path.join(results_dirpath, 'gt')
#     error_save_dir = os.path.join(results_dirpath, 'error')
#     radar_save_dir = os.path.join(results_dirpath, 'radar')
#     semi_save_dir = os.path.join(results_dirpath, 'semi')

#     if not os.path.exists(image_save_dir):
#         os.makedirs(image_save_dir)
#     if not os.path.exists(depth_save_dir):
#         os.makedirs(depth_save_dir)
#     if not os.path.exists(ground_truth_save_dir):
#         os.makedirs(ground_truth_save_dir)
#     if not os.path.exists(error_save_dir):
#         os.makedirs(error_save_dir)
#     if not os.path.exists(radar_save_dir):
#         os.makedirs(radar_save_dir)
#     if not os.path.exists(semi_save_dir):
#         os.makedirs(semi_save_dir)
    
    

#     start_idx = 0

#     image_to_save_name = os.path.join(image_save_dir, 'Image' + str(start_idx+ind) + '.png')
#     pred_depth_to_save_name = os.path.join(depth_save_dir,'Depth' + str(start_idx+ind) + '.png')
#     ground_truth_to_save_name =  os.path.join(ground_truth_save_dir, 'GT' + str(start_idx+ind) + '.png')
#     error_to_save_name = os.path.join(error_save_dir, 'error' + str(start_idx+ind) + '.png')
#     radar_to_save_name = os.path.join(radar_save_dir, 'Radar' + str(start_idx+ind) + '.png')
#     semi_to_save_name = os.path.join(semi_save_dir, 'Semi' + str(start_idx+ind) + '.png')

#     # img shape 확인 및 H, W 추출
#     if img.ndim == 4:  # (B, C, H, W)
#         H, W = img.shape[2], img.shape[3]
#         image_to_process = img[0]
#         image_hwc = image_to_process.transpose(1, 2, 0)
#     elif img.ndim == 3:  # (C, H, W) 또는 (H, W, C)
#         if img.shape[0] == 3:  # (C, H, W)
#             H, W = img.shape[1], img.shape[2]
#             image_hwc = img.transpose(1, 2, 0)
#         else:  # (H, W, C)
#             H, W = img.shape[0], img.shape[1]
#             image_hwc = img
#     elif img.ndim == 2:  # (H, W)
#         H, W = img.shape
#         image_hwc = np.stack([img]*3, axis=-1)
#     else:
#         raise ValueError(f"지원하지 않는 이미지 shape: {img.shape}")

#     # image_rgb = cv2.cvtColor(image_hwc, cv2.COLOR_BGR2RGB) # Sarse-Beats-Dense
#     image_rgb = image_hwc

#     ground_truth = lidar_GT.reshape(H, W)
#     pred = pred.reshape(H, W)

#     # --- Error Map ---
#     validity_map_ground_truth = np.where(ground_truth > 0, 1.0, 0.0)
#     validity_map_output_depth = np.where(pred > 0, 1.0, 0.0)
#     max_error_percent = 0.1
#     error_depth_prediction = np.where(
#         validity_map_ground_truth * validity_map_output_depth,
#         (np.abs(ground_truth - pred) / ground_truth) / max_error_percent,
#         0.0)
#     # --- Error Map ---

#     ground_truth = cv2.dilate(ground_truth, dilation_gt_kernel, iterations=1)
#     error_depth_prediction = cv2.dilate(error_depth_prediction, dilation_kernel, iterations=1)

#     # 1. ground_truth 저장
#     save_image_exact(ground_truth, ground_truth_to_save_name, cmap=color_map, vmin=0.0, vmax=vmax_depth)

#     # 2. image_rgb 저장
#     save_image_exact(image_rgb.astype(np.uint8), image_to_save_name)

#     # 3. pred 저장
#     save_image_exact(pred, pred_depth_to_save_name, cmap=color_map, vmin=0.0, vmax=vmax_depth)

#     # 4. error 저장
#     save_image_exact(error_depth_prediction, error_to_save_name, cmap=color_map, vmin=0.00, vmax=max_error_percent)

#     # 5. input_radar 저장
#     if input_radar is not None:
#         radar_img = input_radar.reshape(H, W)
#         save_image_exact(radar_img, radar_to_save_name, cmap=color_map, vmin=0.0, vmax=vmax_depth)


def visualize_results_batch(vmax_depth, cmap, img_batch, lidar_GT_batch, pred_batch, start_idx, input_radar_batch, dirpath):
    """
    배치 단위로 시각화 결과를 저장하는 함수
    
    Args:
        vmax_depth (float): 최대 깊이 값
        cmap (str): 컬러맵
        img_batch (np.array): 이미지 배치 (B, C, H, W)
        lidar_GT_batch (np.array): LiDAR ground truth 배치 (B, 1, H, W)
        pred_batch (np.array): 예측 결과 배치 (B, 1, H, W)
        start_idx (int): 시작 인덱스
        input_radar_batch (np.array): 레이더 입력 배치 (B, 1, H, W) 또는 None
        dirpath (str): 저장할 디렉토리 경로
    """
    dilation_gt_kernel = np.ones((7, 7), np.uint8)
    dilation_kernel = np.ones((5, 5), np.uint8)

    color_map = plt.get_cmap(cmap)
    
    # --- Visualization ---
    results_dirpath = dirpath

    image_save_dir = os.path.join(results_dirpath, 'image')
    depth_save_dir = os.path.join(results_dirpath, 'output')
    ground_truth_save_dir = os.path.join(results_dirpath, 'gt')
    error_save_dir = os.path.join(results_dirpath, 'error')
    radar_save_dir = os.path.join(results_dirpath, 'radar')
    semi_save_dir = os.path.join(results_dirpath, 'semi')

    # 디렉토리 생성
    for save_dir in [image_save_dir, depth_save_dir, ground_truth_save_dir, error_save_dir, radar_save_dir, semi_save_dir]:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # 배치 크기 확인
    batch_size = img_batch.shape[0]
    
    # 배치 내 각 샘플에 대해 처리
    for i in range(batch_size):
        # 현재 샘플 추출
        img = img_batch[i]  # (C, H, W)
        lidar_GT = lidar_GT_batch[i]  # (1, H, W)
        pred = pred_batch[i]  # (1, H, W)
        input_radar = input_radar_batch[i] if input_radar_batch is not None else None  # (1, H, W) 또는 None
        
        # 파일명 생성
        current_idx = start_idx + i
        image_to_save_name = os.path.join(image_save_dir, f'Image{current_idx}.png')
        pred_depth_to_save_name = os.path.join(depth_save_dir, f'Depth{current_idx}.png')
        ground_truth_to_save_name = os.path.join(ground_truth_save_dir, f'GT{current_idx}.png')
        error_to_save_name = os.path.join(error_save_dir, f'error{current_idx}.png')
        radar_to_save_name = os.path.join(radar_save_dir, f'Radar{current_idx}.png')
        semi_to_save_name = os.path.join(semi_save_dir, f'Semi{current_idx}.png')

        # img shape 확인 및 H, W 추출
        if img.ndim == 3:  # (C, H, W)
            H, W = img.shape[1], img.shape[2]
            image_hwc = img.transpose(1, 2, 0)
        elif img.ndim == 2:  # (H, W)
            H, W = img.shape
            image_hwc = np.stack([img]*3, axis=-1)
        else:
            raise ValueError(f"지원하지 않는 이미지 shape: {img.shape}")

        image_rgb = image_hwc
        # # Normalize image to [0, 1] for display
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        # 역정규화
        image_rgb_denorm = image_rgb * std + mean
        image_rgb_denorm = np.clip(image_rgb_denorm, 0, 1)
        # uint8 변환 후 저장
        image_rgb = image_rgb_denorm * 255.0
        image_rgb = image_rgb.astype(np.uint8)

        # Ground truth와 prediction을 2D로 변환
        ground_truth = lidar_GT.squeeze(0)  # (H, W)
        pred_2d = pred.squeeze(0)  # (H, W)

        # --- Error Map ---
        validity_map_ground_truth = np.where(ground_truth > 0, 1.0, 0.0)
        validity_map_output_depth = np.where(pred_2d > 0, 1.0, 0.0)
        max_error_percent = 0.1
        error_depth_prediction = np.where(
            validity_map_ground_truth * validity_map_output_depth,
            (np.abs(ground_truth - pred_2d) / ground_truth) / max_error_percent,
            0.0)
        # --- Error Map ---

        ground_truth = cv2.dilate(ground_truth, dilation_gt_kernel, iterations=1)
        error_depth_prediction = cv2.dilate(error_depth_prediction, dilation_kernel, iterations=1)
        radar_dilated = cv2.dilate(input_radar.squeeze(0), np.ones((15, 15), np.uint8))

        # 1. ground_truth 저장
        save_image_exact(ground_truth, ground_truth_to_save_name, cmap=color_map, vmin=0.0, vmax=vmax_depth)

        # 2. image_rgb 저장
        save_image_exact(image_rgb, image_to_save_name)

        # 3. pred 저장
        save_image_exact(pred_2d, pred_depth_to_save_name, cmap=color_map, vmin=0.0, vmax=vmax_depth)

        # 4. error 저장
        save_image_exact(error_depth_prediction, error_to_save_name, cmap=color_map, vmin=0.00, vmax=max_error_percent)

        save_image_exact(radar_dilated, radar_to_save_name, cmap=color_map, vmin=0.0, vmax=vmax_depth)
