import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import data_utils

def save_image_exact(data, save_path, dpi=100, cmap='inferno', vmin=0.0, vmax=70.0):
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


def visualize_results():
    dilation_gt_kernel = np.ones((3, 3), np.uint8)
    # dilation_kernel = np.ones((5, 5), np.uint8)

    Stacted_LiDAR_path = '/home/csititan/radar/radar-camera-fusion-depth/testing/nuscenes/nuscenes_test_ground_truth.txt'
    RGB_path = '/home/csititan/radar/radar-camera-fusion-depth/testing/nuscenes/nuscenes_test_image.txt'

    Stacted_LiDAR_path_Paths = data_utils.read_paths(Stacted_LiDAR_path)
    RGB_path_Paths = data_utils.read_paths(RGB_path)

    n_sample = len(Stacted_LiDAR_path_Paths)
    data_format = 'CHW'
    idx = 0
    dirpath = './results_80/'
    stacted_lidar_save_dir = dirpath + 'stacted_lidar/'
    start_idx = 0

    if not os.path.exists(stacted_lidar_save_dir):
        os.makedirs(stacted_lidar_save_dir) 
        
    for idx in tqdm(range(n_sample)):

        depth = data_utils.load_depth(
            Stacted_LiDAR_path_Paths[idx],
            data_format=data_format)

        image = data_utils.load_image(
            RGB_path_Paths[idx],
            normalize=False,
            data_format=data_format)
        
        Stacted_LiDAR = np.squeeze(depth)
        # print(Stacted_LiDAR.shape)

        Stacked_LiDAR_to_save_name =  os.path.join(stacted_lidar_save_dir, 'Stacted_LiDAR' + str(start_idx+idx) + '.png')
        RGB_to_save_name =  os.path.join(stacted_lidar_save_dir, 'RGB' + str(start_idx+idx) + '.png')

        Stacted_LiDAR = cv2.dilate(Stacted_LiDAR, dilation_gt_kernel, iterations=1)

        save_image_exact(Stacted_LiDAR, Stacked_LiDAR_to_save_name, cmap='inferno', vmin=0.0, vmax=80.0)
        save_image_exact(np.transpose(image, (1, 2, 0)).astype(np.uint8), RGB_to_save_name, cmap=None)


def colorbar(filename="_colorbar_jet_80.png"):
    # 값 범위 (0 ~ 70 m)
    vmin, vmax = 0, 80

    # figure와 colorbar용 axis 만들기
    fig, ax = plt.subplots(figsize=(1.0, 6))  # 세로로 긴 colorbar
    fig.subplots_adjust(left=0.5)

    # colormap 설정
    cmap = plt.get_cmap('jet')
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # colorbar 추가
    cb = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        orientation='vertical'
    )

    # 라벨 붙이기
    cb.set_label("Depth (m)")
    cb.ax.yaxis.set_label_position('left')
    cb.ax.tick_params(labelleft=True, labelright=False)

    # 저장
    fig.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)

# visualize_results()
colorbar()