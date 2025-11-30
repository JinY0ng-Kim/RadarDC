import torch
from torch.utils.data import Dataset
import numpy as np
from . import data_utils
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import torchvision.transforms.v2.functional as TF

def apply_augmentations_V7(image, radar, sparse_lidar, interpolated_lidar, prob=0.5):
    """
    Apply data augmentations with given probability
    Args:
        image: RGB image tensor [C, H, W]
        radar: Radar tensor [C, H, W] 
        sparse_lidar: Sparse lidar depth tensor [C, H, W]
        interpolated_lidar: Interpolated lidar depth tensor [C, H, W]
        prob: Probability of applying each augmentation
    Returns:
        Augmented tensors
    """
    # Horizontal flipping
    if random.random() < prob:
        image = F.hflip(image)
        radar = F.hflip(radar)
        sparse_lidar = F.hflip(sparse_lidar)
        interpolated_lidar = F.hflip(interpolated_lidar)
            
    # # Color adjustments (only for RGB image)
    # if random.random() < prob:
    #     # Saturation adjustment
    #     saturation_factor = random.uniform(0.8, 1.2)
    #     image = F.adjust_saturation(image, saturation_factor)
    
    # if random.random() < prob:
    #     # Brightness adjustment
    #     brightness_factor = random.uniform(0.8, 1.2)
    #     image = F.adjust_brightness(image, brightness_factor)
    
    # if random.random() < prob:
    #     # Contrast adjustment
    #     contrast_factor = random.uniform(0.8, 1.2)
    #     image = F.adjust_contrast(image, contrast_factor)
    

    return image, radar, sparse_lidar, interpolated_lidar
    

def apply_augmentations2(image, radar, ground_truth, vp_ground_truth, prob=0.5):
    """
    Apply data augmentations with given probability
    Args:
        image: RGB image tensor [C, H, W]
        radar: Radar tensor [C, H, W] 
        ground_truth: Ground truth depth tensor [C, H, W]
        vp_ground_truth: VP ground truth depth tensor [C, H, W]
        prob: Probability of applying each augmentation
    Returns:
        Augmented tensors
    """
    # Horizontal flipping
    if random.random() < prob:
        image = F.hflip(image)
        radar = F.hflip(radar)
        ground_truth = F.hflip(ground_truth)
        vp_ground_truth = F.hflip(vp_ground_truth)
    
    # Color adjustments (only for RGB image)
    if random.random() < prob:
        # Saturation adjustment
        saturation_factor = random.uniform(0.8, 1.2)
        image = F.adjust_saturation(image, saturation_factor)
    
    if random.random() < prob:
        # Brightness adjustment
        brightness_factor = random.uniform(0.8, 1.2)
        image = F.adjust_brightness(image, brightness_factor)
    
    if random.random() < prob:
        # Contrast adjustment
        contrast_factor = random.uniform(0.8, 1.2)
        image = F.adjust_contrast(image, contrast_factor)
    
    return image, radar, ground_truth, vp_ground_truth


def apply_augmentations(image, radar, ground_truth, K_cam=None, prob=0.5):
    """
    Apply data augmentations with given probability
    Args:
        image: RGB image tensor [C, H, W]
        radar: Radar tensor [C, H, W] 
        ground_truth: Ground truth depth tensor [C, H, W]
        K_cam: Camera intrinsic matrix [3, 3] (optional)
        prob: Probability of applying each augmentation
    Returns:
        Augmented tensors
    """
    # Horizontal flipping
    if random.random() < prob:
        image = F.hflip(image)
        radar = F.hflip(radar)
        ground_truth = F.hflip(ground_truth)
        
        # Adjust camera intrinsic matrix for horizontal flip
        if K_cam is not None:
            K_cam = K_cam.clone()
            _, _, W = image.shape
            # cx_new = W - cx
            K_cam[0, 2] = W - K_cam[0, 2]
    
    # # Color adjustments (only for RGB image)
    # if random.random() < prob:
    #     # Saturation adjustment
    #     saturation_factor = random.uniform(0.8, 1.2)
    #     image = F.adjust_saturation(image, saturation_factor)
    
    # if random.random() < prob:
    #     # Brightness adjustment
    #     brightness_factor = random.uniform(0.8, 1.2)
    #     image = F.adjust_brightness(image, brightness_factor)
    
    # if random.random() < prob:
    #     # Contrast adjustment
    #     contrast_factor = random.uniform(0.8, 1.2)
    #     image = F.adjust_contrast(image, contrast_factor)
    
    if K_cam is not None:
        return image, radar, ground_truth, K_cam
    else:
        return image, radar, ground_truth
    
class nuScenesDataset_RadarDC_V7(torch.utils.data.Dataset):
    def __init__(self,
                 image_paths,
                 radar_paths,
                 sparse_lidar_paths,
                 interpolated_lidar_paths,
                 mode='train',
                 total_points_sampled=40,
                 sample_probability_of_lidar=0.10):
        

        self.image_paths = data_utils.read_paths(image_paths)
        self.radar_paths = data_utils.read_paths(radar_paths)
        self.sparse_lidar_paths = data_utils.read_paths(sparse_lidar_paths)
        self.interpolated_lidar_paths = data_utils.read_paths(interpolated_lidar_paths)
        self.mode = mode

        self.IMAGENET_DATASET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_DATASET_STD = (0.229, 0.224, 0.225)

        
        # 패치 관련 설정 제거

        self.data_format = 'CHW'
        self.total_points_sampled = total_points_sampled
        self.sample_probability_of_lidar = sample_probability_of_lidar

        self.n_sample = len(self.image_paths)

        for paths in [self.sparse_lidar_paths, self.interpolated_lidar_paths, self.radar_paths]:
            assert len(paths) == self.n_sample

        
        
    
    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        
        # Load image
        image = data_utils.load_image(
            self.image_paths[index],
            normalize=True,
            data_format=self.data_format)
        _, H, W = image.shape

        
        # Load radar
        radar_points = np.load(self.radar_paths[index])

        if radar_points.ndim == 1:
            # Expand to 1 x 3
            radar_points = np.expand_dims(radar_points, axis=0)
        
        # randomly sample radar points to output
        if radar_points.shape[0] <= self.total_points_sampled:
            radar_points = np.repeat(radar_points, 100, axis=0)
        random_idx = np.random.randint(radar_points.shape[0], size=self.total_points_sampled)
        radar_points = radar_points[random_idx, :]
        
        # Load gt
        sparse_lidar = data_utils.load_depth(
            self.sparse_lidar_paths[index],
            data_format=self.data_format)

        interpolated_lidar = data_utils.load_depth(
            self.interpolated_lidar_paths[index],
            data_format=self.data_format)
        
        if random.random() < self.sample_probability_of_lidar:

            ground_truth_for_sampling = np.copy(interpolated_lidar)
            ground_truth_for_sampling = ground_truth_for_sampling.squeeze()
            # Find lidar points with depth greater than 1
            idx_lidar_samples = np.where(ground_truth_for_sampling > 1)

            # randomly sample total_points_sampled number of points from the lidar
            random_indices = random.sample(range(0, len(idx_lidar_samples[0])), self.total_points_sampled)

            points_x = idx_lidar_samples[1][random_indices]
            points_y = idx_lidar_samples[0][random_indices]
            points_z = ground_truth_for_sampling[points_y, points_x]

            noise_for_fake_radar_x = np.random.normal(0,25,radar_points.shape[0])
            noise_for_fake_radar_z = np.random.uniform(low=0.0, high=0.4, size=radar_points.shape[0])

            fake_radar_points = np.copy(radar_points)
            fake_radar_points[:,0] = points_x + noise_for_fake_radar_x
            fake_radar_points[:,0] = np.clip(fake_radar_points[:,0], 0, ground_truth_for_sampling.shape[1])
            fake_radar_points[:,2] = points_z + noise_for_fake_radar_z
            # we keep the y as the same it is since it is erroneous

            # convert x and y indices back to int after adding noise
            fake_radar_points[:,0] = fake_radar_points[:,0].astype(int)
            fake_radar_points[:,1] = fake_radar_points[:,1].astype(int)

            radar_points = np.copy(fake_radar_points)

        image, radar_points, sparse_lidar, interpolated_lidar = [
            T.astype(np.float32)
            for T in [image, radar_points, sparse_lidar, interpolated_lidar]
        ]
        
        _, H, W = image.shape
        radar_depth_map = np.zeros((H, W), dtype=np.float32)

        for pt in radar_points:
            x, y, z = int(round(pt[0])), int(round(pt[1])), pt[2]
            if 0 <= x < W and 0 <= y < H:
                radar_depth_map[y, x] = z
        radar_depth_map = np.expand_dims(radar_depth_map, axis=0)

        radar_ori = data_utils.load_radar_npy(
            self.radar_paths[index],
            image_shape=(H, W),
            data_format=self.data_format)


        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if isinstance(sparse_lidar, np.ndarray):
            sparse_lidar = torch.from_numpy(sparse_lidar)
        if isinstance(interpolated_lidar, np.ndarray):
            interpolated_lidar = torch.from_numpy(interpolated_lidar)

        # if isinstance(radar_depth_map, np.ndarray):
        #     radar_depth_map = torch.from_numpy(radar_depth_map)
        if isinstance(radar_ori, np.ndarray):
            radar_ori = torch.from_numpy(radar_ori)

        image = TF.normalize(
                image,
                mean=self.IMAGENET_DATASET_MEAN,
                std=self.IMAGENET_DATASET_STD,
        )

        # Apply data augmentations only during training
        if self.mode == 'train':
            image, radar_depth_map, sparse_lidar, interpolated_lidar = apply_augmentations_V7(image, radar_ori, sparse_lidar, interpolated_lidar, prob=0.5)

        return image, radar_depth_map, sparse_lidar, interpolated_lidar


class nuScenesDataset_RadarDC_V6(torch.utils.data.Dataset):
    def __init__(self,
                 image_paths,
                 radar_paths,
                 ground_truth_paths,
                 K_cam_paths,
                 mode='train',
                 total_points_sampled=40,
                 sample_probability_of_lidar=0.10):
        

        self.image_paths = data_utils.read_paths(image_paths)
        self.radar_paths = data_utils.read_paths(radar_paths)
        self.ground_truth_paths = data_utils.read_paths(ground_truth_paths)
        self.K_cam_paths = data_utils.read_paths(K_cam_paths)
        self.mode = mode

        self.IMAGENET_DATASET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_DATASET_STD = (0.229, 0.224, 0.225)

        
        # 패치 관련 설정 제거

        self.data_format = 'CHW'
        self.total_points_sampled = total_points_sampled
        self.sample_probability_of_lidar = sample_probability_of_lidar

        self.n_sample = len(self.image_paths)

        for paths in [self.ground_truth_paths, self.radar_paths, self.K_cam_paths]:
            assert len(paths) == self.n_sample

        
        
    
    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        
        # Load image
        image = data_utils.load_image(
            self.image_paths[index],
            normalize=True,
            data_format=self.data_format)
        _, H, W = image.shape

        
        # Load radar
        radar_points = np.load(self.radar_paths[index])

        if radar_points.ndim == 1:
            # Expand to 1 x 3
            radar_points = np.expand_dims(radar_points, axis=0)
        
        # randomly sample radar points to output
        if radar_points.shape[0] <= self.total_points_sampled:
            radar_points = np.repeat(radar_points, 100, axis=0)
        random_idx = np.random.randint(radar_points.shape[0], size=self.total_points_sampled)
        radar_points = radar_points[random_idx, :]
        
        # Load gt
        ground_truth = data_utils.load_depth(
            self.ground_truth_paths[index],
            data_format=self.data_format)

        # Load K_cam
        K_cam = np.load(self.K_cam_paths[index])
        K_cam = K_cam.astype(np.float32)

        if random.random() < self.sample_probability_of_lidar:

            ground_truth_for_sampling = np.copy(ground_truth)
            ground_truth_for_sampling = ground_truth_for_sampling.squeeze()
            # Find lidar points with depth greater than 1
            idx_lidar_samples = np.where(ground_truth_for_sampling > 1)

            # randomly sample total_points_sampled number of points from the lidar
            random_indices = random.sample(range(0, len(idx_lidar_samples[0])), self.total_points_sampled)

            points_x = idx_lidar_samples[1][random_indices]
            points_y = idx_lidar_samples[0][random_indices]
            points_z = ground_truth_for_sampling[points_y, points_x]

            noise_for_fake_radar_x = np.random.normal(0,25,radar_points.shape[0])
            noise_for_fake_radar_z = np.random.uniform(low=0.0, high=0.4, size=radar_points.shape[0])

            fake_radar_points = np.copy(radar_points)
            fake_radar_points[:,0] = points_x + noise_for_fake_radar_x
            fake_radar_points[:,0] = np.clip(fake_radar_points[:,0], 0, ground_truth_for_sampling.shape[1])
            fake_radar_points[:,2] = points_z + noise_for_fake_radar_z
            # we keep the y as the same it is since it is erroneous

            # convert x and y indices back to int after adding noise
            fake_radar_points[:,0] = fake_radar_points[:,0].astype(int)
            fake_radar_points[:,1] = fake_radar_points[:,1].astype(int)

            radar_points = np.copy(fake_radar_points)

        image, radar_points, ground_truth = [
            T.astype(np.float32)
            for T in [image, radar_points, ground_truth]
        ]
        
        _, H, W = image.shape
        radar_depth_map = np.zeros((H, W), dtype=np.float32)

        for pt in radar_points:
            x, y, z = int(round(pt[0])), int(round(pt[1])), pt[2]
            if 0 <= x < W and 0 <= y < H:
                radar_depth_map[y, x] = z
        radar_depth_map = np.expand_dims(radar_depth_map, axis=0)

        radar_ori = data_utils.load_radar_npy(
            self.radar_paths[index],
            image_shape=(H, W),
            data_format=self.data_format)


        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if isinstance(ground_truth, np.ndarray):
            ground_truth = torch.from_numpy(ground_truth)
        if isinstance(radar_depth_map, np.ndarray):
            radar_depth_map = torch.from_numpy(radar_depth_map)
        if isinstance(K_cam, np.ndarray):
            K_cam = torch.from_numpy(K_cam)
        if isinstance(radar_ori, np.ndarray):
            radar_ori = torch.from_numpy(radar_ori)

        image = TF.normalize(
                image,
                mean=self.IMAGENET_DATASET_MEAN,
                std=self.IMAGENET_DATASET_STD,
        )

        # Apply data augmentations only during training
        if self.mode == 'train':
            image, radar_depth_map, ground_truth, K_cam = apply_augmentations(image, radar_ori, ground_truth, K_cam, prob=0.5)

        return image, radar_depth_map, ground_truth, K_cam


class nuScenesDataset_RadarDC_VP(torch.utils.data.Dataset):
    def __init__(self,
                 image_paths,
                 radar_paths,
                 vp_ground_truth_paths,
                 mode='train',
                 total_points_sampled=40,
                 sample_probability_of_lidar=0.10):
        

        self.image_paths = data_utils.read_paths(image_paths)
        self.radar_paths = data_utils.read_paths(radar_paths)
        self.vp_ground_truth_paths = data_utils.read_paths(vp_ground_truth_paths)
        self.mode = mode

        self.IMAGENET_DATASET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_DATASET_STD = (0.229, 0.224, 0.225)

        self.data_format = 'CHW'
        self.total_points_sampled = total_points_sampled
        self.sample_probability_of_lidar = sample_probability_of_lidar

        self.n_sample = len(self.image_paths)

        for paths in [self.vp_ground_truth_paths, self.radar_paths]:
            assert len(paths) == self.n_sample
   
    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        
        # Load image
        image = data_utils.load_image(
            self.image_paths[index],
            normalize=True,
            data_format=self.data_format)
        _, H, W = image.shape

        
        # Load radar
        radar_points = np.load(self.radar_paths[index])

        if radar_points.ndim == 1:
            # Expand to 1 x 3
            radar_points = np.expand_dims(radar_points, axis=0)
        
        # randomly sample radar points to output
        if radar_points.shape[0] <= self.total_points_sampled:
            radar_points = np.repeat(radar_points, 100, axis=0)
        random_idx = np.random.randint(radar_points.shape[0], size=self.total_points_sampled)
        radar_points = radar_points[random_idx, :]
        
        # Load gt
        vp_ground_truth = data_utils.load_depth(
            self.vp_ground_truth_paths[index],
            data_format=self.data_format)

        if random.random() < self.sample_probability_of_lidar:

            ground_truth_for_sampling = np.copy(vp_ground_truth)
            ground_truth_for_sampling = ground_truth_for_sampling.squeeze()
            # Find lidar points with depth greater than 1
            idx_lidar_samples = np.where(ground_truth_for_sampling > 1)

            # randomly sample total_points_sampled number of points from the lidar
            random_indices = random.sample(range(0, len(idx_lidar_samples[0])), self.total_points_sampled)

            points_x = idx_lidar_samples[1][random_indices]
            points_y = idx_lidar_samples[0][random_indices]
            points_z = ground_truth_for_sampling[points_y, points_x]

            noise_for_fake_radar_x = np.random.normal(0,25,radar_points.shape[0])
            noise_for_fake_radar_z = np.random.uniform(low=0.0, high=0.4, size=radar_points.shape[0])

            fake_radar_points = np.copy(radar_points)
            fake_radar_points[:,0] = points_x + noise_for_fake_radar_x
            fake_radar_points[:,0] = np.clip(fake_radar_points[:,0], 0, ground_truth_for_sampling.shape[1])
            fake_radar_points[:,2] = points_z + noise_for_fake_radar_z
            # we keep the y as the same it is since it is erroneous

            # convert x and y indices back to int after adding noise
            fake_radar_points[:,0] = fake_radar_points[:,0].astype(int)
            fake_radar_points[:,1] = fake_radar_points[:,1].astype(int)

            radar_points = np.copy(fake_radar_points)

        image, radar_points, vp_ground_truth = [
            T.astype(np.float32)
            for T in [image, radar_points, vp_ground_truth]
        ]
        
        _, H, W = image.shape
        radar_depth_map = np.zeros((H, W), dtype=np.float32)

        for pt in radar_points:
            x, y, z = int(round(pt[0])), int(round(pt[1])), pt[2]
            if 0 <= x < W and 0 <= y < H:
                radar_depth_map[y, x] = z
        radar_depth_map = np.expand_dims(radar_depth_map, axis=0)


        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if isinstance(vp_ground_truth, np.ndarray):
            vp_ground_truth = torch.from_numpy(vp_ground_truth)
        if isinstance(radar_depth_map, np.ndarray):
            radar_depth_map = torch.from_numpy(radar_depth_map)

        image = TF.normalize(
                image,
                mean=self.IMAGENET_DATASET_MEAN,
                std=self.IMAGENET_DATASET_STD,
        )

        return image, radar_depth_map, vp_ground_truth

class nuScenesDataset_RadarDC(torch.utils.data.Dataset):
    def __init__(self,
                 image_paths,
                 radar_paths,
                 ground_truth_paths,
                 mode='train',
                 total_points_sampled=40,
                 sample_probability_of_lidar=0.10):
        

        self.image_paths = data_utils.read_paths(image_paths)
        self.radar_paths = data_utils.read_paths(radar_paths)
        self.ground_truth_paths = data_utils.read_paths(ground_truth_paths)
        self.mode = mode

        self.IMAGENET_DATASET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_DATASET_STD = (0.229, 0.224, 0.225)

        
        # 패치 관련 설정 제거

        self.data_format = 'CHW'
        self.total_points_sampled = total_points_sampled
        self.sample_probability_of_lidar = sample_probability_of_lidar

        self.n_sample = len(self.image_paths)

        for paths in [self.ground_truth_paths, self.radar_paths]:
            assert len(paths) == self.n_sample

        
        
    
    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        
        # Load image
        image = data_utils.load_image(
            self.image_paths[index],
            normalize=True,
            data_format=self.data_format)
        _, H, W = image.shape

        
        # Load radar
        radar_points = np.load(self.radar_paths[index])

        if radar_points.ndim == 1:
            # Expand to 1 x 3
            radar_points = np.expand_dims(radar_points, axis=0)
        
        # randomly sample radar points to output
        if radar_points.shape[0] <= self.total_points_sampled:
            radar_points = np.repeat(radar_points, 100, axis=0)
        random_idx = np.random.randint(radar_points.shape[0], size=self.total_points_sampled)
        radar_points = radar_points[random_idx, :]
        
        # Load gt
        ground_truth = data_utils.load_depth(
            self.ground_truth_paths[index],
            data_format=self.data_format)

        if random.random() < self.sample_probability_of_lidar:

            ground_truth_for_sampling = np.copy(ground_truth)
            ground_truth_for_sampling = ground_truth_for_sampling.squeeze()
            # Find lidar points with depth greater than 1
            idx_lidar_samples = np.where(ground_truth_for_sampling > 1)

            # randomly sample total_points_sampled number of points from the lidar
            random_indices = random.sample(range(0, len(idx_lidar_samples[0])), self.total_points_sampled)

            points_x = idx_lidar_samples[1][random_indices]
            points_y = idx_lidar_samples[0][random_indices]
            points_z = ground_truth_for_sampling[points_y, points_x]

            noise_for_fake_radar_x = np.random.normal(0,25,radar_points.shape[0])
            noise_for_fake_radar_z = np.random.uniform(low=0.0, high=0.4, size=radar_points.shape[0])

            fake_radar_points = np.copy(radar_points)
            fake_radar_points[:,0] = points_x + noise_for_fake_radar_x
            fake_radar_points[:,0] = np.clip(fake_radar_points[:,0], 0, ground_truth_for_sampling.shape[1])
            fake_radar_points[:,2] = points_z + noise_for_fake_radar_z
            # we keep the y as the same it is since it is erroneous

            # convert x and y indices back to int after adding noise
            fake_radar_points[:,0] = fake_radar_points[:,0].astype(int)
            fake_radar_points[:,1] = fake_radar_points[:,1].astype(int)

            radar_points = np.copy(fake_radar_points)

        image, radar_points, ground_truth = [
            T.astype(np.float32)
            for T in [image, radar_points, ground_truth]
        ]
        
        _, H, W = image.shape
        radar_depth_map = np.zeros((H, W), dtype=np.float32)

        for pt in radar_points:
            x, y, z = int(round(pt[0])), int(round(pt[1])), pt[2]
            if 0 <= x < W and 0 <= y < H:
                radar_depth_map[y, x] = z
        radar_depth_map = np.expand_dims(radar_depth_map, axis=0)


        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if isinstance(ground_truth, np.ndarray):
            ground_truth = torch.from_numpy(ground_truth)
        if isinstance(radar_depth_map, np.ndarray):
            radar_depth_map = torch.from_numpy(radar_depth_map)

        image = TF.normalize(
                image,
                mean=self.IMAGENET_DATASET_MEAN,
                std=self.IMAGENET_DATASET_STD,
        )

        # Apply data augmentations only during training
        if self.mode == 'train':
            image, radar_depth_map, ground_truth = apply_augmentations(image, radar_depth_map, ground_truth, prob=0.5)

        return image, radar_depth_map, ground_truth


class nuScenesDataset_UniDepth(torch.utils.data.Dataset):
    def __init__(self,
                 image_paths,
                 ground_truth_paths ):
        

        self.image_paths = data_utils.read_paths(image_paths)
        self.ground_truth_paths = data_utils.read_paths(ground_truth_paths)

        self.n_sample = len(self.image_paths)

        for paths in [self.ground_truth_paths]:
            assert len(paths) == self.n_sample

        self.data_format = 'CHW'
        
    
    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        
        # Load image
        image = data_utils.load_image(
            self.image_paths[index],
            normalize=False,
            data_format=self.data_format)
        _, H, W = image.shape
        
        # Load gt
        ground_truth = data_utils.load_depth(
            self.ground_truth_paths[index],
            data_format=self.data_format)
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if isinstance(ground_truth, np.ndarray):
            ground_truth = torch.from_numpy(ground_truth)

        return image, ground_truth
    
class nuScenesDataset_BPNet(torch.utils.data.Dataset):
    def __init__(self,
                 image_paths,
                 depth_paths,
                 ground_truth_paths,
                 K_cam_paths,
                 mode=None,
                 **kwargs):
        
        
        self.mode = mode

        self.image_paths = data_utils.read_paths(image_paths)
        self.depth_paths = data_utils.read_paths(depth_paths)
        self.ground_truth_paths = data_utils.read_paths(ground_truth_paths)
        self.K_cam_paths = data_utils.read_paths(K_cam_paths)

        self.n_sample = len(self.image_paths)

        for paths in [self.depth_paths, self.ground_truth_paths]:
            assert len(paths) == self.n_sample

        self.data_format = 'CHW'
        
    
    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        
        # Load image
        image = data_utils.load_image(
            self.image_paths[index],
            normalize=False,
            data_format=self.data_format)
        _, H, W = image.shape

        # Load depth
        # radar_depth = data_utils.load_depth(
        radar_depth = data_utils.load_radar_npy(
            self.depth_paths[index],
            image_shape=(H, W),
            data_format=self.data_format)
        
        # Load gt
        ground_truth = data_utils.load_depth(
            self.ground_truth_paths[index],
            data_format=self.data_format)
        
        # Load K_cam
        K_cam = np.load(self.K_cam_paths[index])
        K_cam = K_cam.astype(np.float32)

        image, radar_depth, ground_truth, K_cam = self.to_torch(image, radar_depth, ground_truth, K_cam)

        # 입력 데이터 크기를 32의 배수로 맞추기 위한 전처리
        # (H, W) = (900, 1600) -> (896, 1600)
        # 896 = 32 * 28
        target_height, target_width = 896, 1600
        
        transform_img = T.Compose([
            T.CenterCrop((target_height, target_width)),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_depth = T.CenterCrop((target_height, target_width))
        
        image = transform_img(image)
        radar_depth = transform_depth(radar_depth)
        ground_truth = transform_depth(ground_truth)

        # sample = {'rgb': image, 'radar_depth': radar_depth, 'lidar_gt': ground_truth, 'K_cam': K_cam}
        sample = image, radar_depth, K_cam, ground_truth


        # for key, value in sample.items():
        #         if isinstance(value, np.ndarray):
        #             sample[key] = value.astype(np.float32)
        #         else:
        #             sample[key] = value.to(torch.float32)

        # return sample
        return image, radar_depth, K_cam, ground_truth
    
    def to_torch(self, *args):
        """Converts numpy arrays to PyTorch tensors."""
        tensors = []
        for item in args:
            if isinstance(item, np.ndarray):
                tensor = torch.from_numpy(item)
            else: # Assuming it's already a tensor or similar
                tensor = item
            tensors.append(tensor.float())
        return tensors

class nuScenesDataset_DepthPrompt(torch.utils.data.Dataset):
    def __init__(self,
                 image_paths,
                 depth_paths,
                 ground_truth_paths):

        self.n_sample = len(image_paths)

        self.image_paths = image_paths
        self.depth_paths = depth_paths


        if ground_truth_paths is not None and None not in ground_truth_paths:
            assert self.n_sample == len(ground_truth_paths)
            self.ground_truth_available = True
        else:
            self.ground_truth_available = False

        self.ground_truth_paths = ground_truth_paths

        for paths in [depth_paths, ground_truth_paths]:
            assert len(paths) == self.n_sample

        self.data_format = 'CHW'

    def __getitem__(self, index):
        
        # Load image
        image = data_utils.load_image(
            self.image_paths[index],
            normalize=False,
            data_format=self.data_format)
        _, H, W = image.shape

        # Load depth
        # radar_depth = data_utils.load_depth(
        radar_depth = data_utils.load_radar_npy(
            self.depth_paths[index],
            image_shape=(H, W),
            data_format=self.data_format)
        
        # Load gt
        ground_truth = data_utils.load_depth(
                self.ground_truth_paths[index],
                data_format=self.data_format)
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if isinstance(radar_depth, np.ndarray):
            radar_depth = torch.from_numpy(radar_depth)
        if isinstance(ground_truth, np.ndarray):
            ground_truth = torch.from_numpy(ground_truth)

        radar_depth = radar_depth.float()

        normalize_only = T.Compose([
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # normalize_crop = T.Compose([
        #     T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #     T.CenterCrop((352, 1600)),
        #     T.Resize((240, 1216), interpolation=Image.BILINEAR)
        # ])

        # center_crop_dep = T.Compose([
        #      T.CenterCrop((352, 1600)),
        #      T.Resize((240, 1216), interpolation=Image.NEAREST)
        #  ])   
        
        image = normalize_only(image)
        # depth = center_crop_dep(depth)

        output = {'rgb': image, 'dep': radar_depth, 'gt': ground_truth, 'K': radar_depth, 'rgb_480640':radar_depth, 'dep_480640':radar_depth, 'num_sample':radar_depth}


        # for key, value in sample.items():
        #         if isinstance(value, np.ndarray):
        #             sample[key] = value.astype(np.float32)
        #         else:
        #             sample[key] = value.to(torch.float32)

        return output

    def __len__(self):
        return self.n_sample
