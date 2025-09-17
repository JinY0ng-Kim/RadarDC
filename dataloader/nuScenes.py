import torch
from torch.utils.data import Dataset
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from pyquaternion import Quaternion
from PIL import Image
import os

class RadarCameraSparseDepthDataset(Dataset):
    def __init__(self, dataroot, version="v1.0-trainval", camera="CAM_FRONT", image_size=(1600, 900), transform=None):
        """
        image_size: (width, height) 모델 입력 크기
        """
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.camera = camera
        self.transform = transform
        self.image_size = image_size  # (W,H)
        # camera sample_data token 리스트
        self.cam_tokens = [
            sd['token'] for sd in self.nusc.sample_data 
            if sd['sensor_modality']=='camera' and self.camera in sd['filename']
        ]
        
    def __len__(self):
        return len(self.cam_tokens)
    
    def __getitem__(self, idx):
        cam_sd_token = self.cam_tokens[idx]
        cam_sd_rec = self.nusc.get('sample_data', cam_sd_token)
        sample = self.nusc.get('sample', cam_sd_rec['sample_token'])
        
        # radar sample_data 가져오기
        radar_sd_rec = self.nusc.get('sample_data', sample['data']['RADAR_FRONT'])
        radar_file = os.path.join(self.nusc.dataroot, radar_sd_rec['filename'])
        rpc = RadarPointCloud.from_file(radar_file)
        radar_pts = rpc.points[:3,:]  # xyz
        
        # 좌표 변환
        pts_cam = self.transform_radar_to_cam(radar_pts, radar_sd_rec, cam_sd_rec)
        
        # projection
        cs_cam = self.nusc.get('calibrated_sensor', cam_sd_rec['calibrated_sensor_token'])
        K = np.array(cs_cam['camera_intrinsic'])
        mask = pts_cam[2,:] > 0
        pts_cam = pts_cam[:, mask]
        uv = K @ pts_cam
        uv = uv[:2] / uv[2]
        depth = pts_cam[2, :]
        
        # 이미지 로드 및 resize
        img_path = os.path.join(self.nusc.dataroot, cam_sd_rec['filename'])
        img = np.array(Image.open(img_path).convert('RGB'))
        H_img, W_img = img.shape[:2]
        W_target, H_target = self.image_size
        scale_x = W_target / W_img
        scale_y = H_target / H_img
        img_resized = np.array(Image.fromarray(img).resize((W_target,H_target)))
        
        # Sparse depth map 생성
        depth_map = np.zeros(self.image_size[::-1], dtype=np.float32)  # HxW
        u = (uv[0] * scale_x).astype(int)
        v = (uv[1] * scale_y).astype(int)
        valid = (u>=0) & (u<W_target) & (v>=0) & (v<H_target)
        u = u[valid]
        v = v[valid]
        depth = depth[valid]
        depth_map[v,u] = depth
        
        # PyTorch Tensor 변환 (C,H,W)
        img_tensor = torch.from_numpy(img_resized).permute(2,0,1).float() / 255.0
        depth_tensor = torch.from_numpy(depth_map).unsqueeze(0)
        
        return {
            "image": img_tensor,
            "sparse_depth": depth_tensor
        }
    
    def transform_radar_to_cam(self, pts, radar_sd, cam_sd):
        # Radar -> Ego -> Global -> Ego_Cam -> Camera Sensor
        cs_radar = self.nusc.get('calibrated_sensor', radar_sd['calibrated_sensor_token'])
        pts = Quaternion(cs_radar['rotation']).rotation_matrix @ pts + np.array(cs_radar['translation']).reshape(3,1)
        ego_radar = self.nusc.get('ego_pose', radar_sd['ego_pose_token'])
        pts = Quaternion(ego_radar['rotation']).rotation_matrix @ pts + np.array(ego_radar['translation']).reshape(3,1)
        ego_cam = self.nusc.get('ego_pose', cam_sd['ego_pose_token'])
        pts = Quaternion(ego_cam['rotation']).rotation_matrix.T @ (pts - np.array(ego_cam['translation']).reshape(3,1))
        cs_cam = self.nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        pts = Quaternion(cs_cam['rotation']).rotation_matrix.T @ (pts - np.array(cs_cam['translation']).reshape(3,1))
        return pts
