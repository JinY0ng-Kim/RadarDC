from nuscenes.nuscenes import NuScenes, NuScenesExplorer
import os, sys, argparse
import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud
import pickle
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
import multiprocessing as mp

sys.path.insert(0, 'src')
from _my_utils import data_utils

MAX_SCENES = 150


'''
Output filepaths
'''
TRAIN_REF_DIRPATH = os.path.join('training', 'nuscenes')
VAL_REF_DIRPATH = os.path.join('validation', 'nuscenes')
TEST_REF_DIRPATH = os.path.join('testing', 'nuscenes')

TEST_IMAGE_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_image.txt')
TEST_LIDAR_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_lidar.txt')
TEST_RADAR_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_radar.txt')


'''
Set up input arguments
'''
parser = argparse.ArgumentParser()

parser.add_argument('--nuscenes_data_root_dirpath',
    type=str, required=True, help='Path to nuscenes dataset')
parser.add_argument('--nuscenes_data_derived_dirpath',
    type=str, required=True, help='Path to derived dataset')
parser.add_argument('--n_scenes_to_process',
    type=int, default=MAX_SCENES, help='Number of scenes to process')
parser.add_argument('--paths_only',
    action='store_true', help='If set, then only produce paths')
parser.add_argument('--n_thread',
    type=int, default=40, help='Number of threads to use in parallel pool')
parser.add_argument('--debug',
    action='store_true', help='If set, then enter debug mode')


args = parser.parse_args()


# Create global nuScene object
nusc = NuScenes(
    version='v1.0-test',
    dataroot=args.nuscenes_data_root_dirpath,
    verbose=True)

nusc_explorer = NuScenesExplorer(nusc)

def point_cloud_to_image(nusc,
                         point_cloud,
                         lidar_sensor_token,
                         camera_token,
                         min_distance_from_camera=1.0):
    '''
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.

    Arg(s):
        nusc : Object
            nuScenes data object
        point_cloud : PointCloud
            nuScenes point cloud object
        lidar_sensor_token : str
            token to access lidar data in nuscenes sample_data object
        camera_token : str
            token to access camera data in nuscenes sample_data object
        minimum_distance_from_camera : float32
            threshold for removing points that exceeds minimum distance from camera
    Returns:
        numpy[float32] : 3 x N array of x, y, z
        numpy[float32] : N array of z
        numpy[float32] : camera image
    '''

    # Get dictionary of containing path to image, pose, etc.
    camera = nusc.get('sample_data', camera_token)
    lidar_sensor = nusc.get('sample_data', lidar_sensor_token)

    image_path = os.path.join(nusc.dataroot, camera['filename'])
    image = data_utils.load_image(image_path)

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    pose_lidar_to_body = nusc.get('calibrated_sensor', lidar_sensor['calibrated_sensor_token'])
    point_cloud.rotate(Quaternion(pose_lidar_to_body['rotation']).rotation_matrix)
    point_cloud.translate(np.array(pose_lidar_to_body['translation']))

    # Second step: transform from ego to the global frame.
    pose_body_to_global = nusc.get('ego_pose', lidar_sensor['ego_pose_token'])
    point_cloud.rotate(Quaternion(pose_body_to_global['rotation']).rotation_matrix)
    point_cloud.translate(np.array(pose_body_to_global['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    pose_body_to_global = nusc.get('ego_pose', camera['ego_pose_token'])
    point_cloud.translate(-np.array(pose_body_to_global['translation']))
    point_cloud.rotate(Quaternion(pose_body_to_global['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    pose_body_to_camera = nusc.get('calibrated_sensor', camera['calibrated_sensor_token'])
    point_cloud.translate(-np.array(pose_body_to_camera['translation']))
    point_cloud.rotate(Quaternion(pose_body_to_camera['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depth = point_cloud.points[2, :]

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    # Points will be 3 x N
    points = view_points(point_cloud.points[:3, :], np.array(pose_body_to_camera['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depth.shape[0], dtype=bool)
    mask = np.logical_and(mask, depth > min_distance_from_camera)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < image.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < image.shape[0] - 1)

    # Select points that are more than min distance from camera and not on edge of image
    points = points[:, mask]
    depth = depth[mask]

    return points, depth, image



def get_radar_points(nusc, nusc_explorer, current_sample_token):
    '''
    Get radar points from current sample and project them to camera frame.

    Arg(s):
        nusc : NuScenes Object
            nuScenes object instance
        nusc_explorer : NuScenesExplorer Object
            nuScenes explorer object instance
        current_sample_token : str
            token for accessing the current sample data
    Returns:
        numpy[float32] : 2 x N of x, y for radar points projected into the image
        numpy[float32] : N depths of radar points
    '''

    # Get the sample
    current_sample = nusc.get('sample', current_sample_token)

    # Get radar token in the current sample
    radar_token = current_sample['data']['RADAR_FRONT']

    # Get the camera token for the current sample
    camera_token = current_sample['data']['CAM_FRONT']

    # Project the radar frame into the camera frame
    points_radar, depth_radar, _ = nusc_explorer.map_pointcloud_to_image(
        pointsensor_token=radar_token,
        camera_token=camera_token)

    return points_radar, depth_radar

def lidar_depth_map_from_token(nusc,
                               nusc_explorer,
                               current_sample_token):
    '''
    Picks current_sample_token as reference and projects lidar points onto the image plane.

    Arg(s):
        nusc : NuScenes Object
            nuScenes object instance
        nusc_explorer : NuScenesExplorer Object
            nuScenes explorer object instance
        current_sample_token : str
            token for accessing the current sample data
    Returns:
        numpy[float32] : H x W depth
    '''

    current_sample = nusc.get('sample', current_sample_token)
    lidar_token = current_sample['data']['LIDAR_TOP']
    main_camera_token = current_sample['data']['CAM_FRONT']

    # project the lidar frame into the camera frame
    main_points_lidar, main_depth_lidar, main_image = nusc_explorer.map_pointcloud_to_image(
        pointsensor_token=lidar_token,
        camera_token=main_camera_token)

    depth_map = points_to_depth_map(main_points_lidar, main_depth_lidar, main_image)

    return depth_map

def points_to_depth_map(points, depth, image):
    '''
    Plots the depth values onto the image plane

    Arg(s):
        points : numpy[float32]
            2 x N matrix in x, y
        depth : numpy[float32]
            N scales for z
        image : numpy[float32]
            H x W x 3 image for reference frame size
    Returns:
        numpy[float32] : H x W image with depth plotted
    '''

    # Plot points onto the image
    image = np.asarray(image)
    depth_map = np.zeros((image.shape[0], image.shape[1]))

    points_quantized = np.round(points).astype(int)

    for pt_idx in range(0, points_quantized.shape[1]):
        x = points_quantized[0, pt_idx]
        y = points_quantized[1, pt_idx]
        depth_map[y, x] = depth[pt_idx]

    return depth_map

def process_scene(args):
    '''
    Processes one scene from first sample to last sample

    Arg(s):
        args : tuple(Object, Object, str, int, str, str, int, int, str, bool)
            nusc : NuScenes Object
                nuScenes object instance
            nusc_explorer : NuScenesExplorer Object
                nuScenes explorer object instance
            tag : str
                train, val
            scene_id : int
                identifier for one scene
            first_sample_token : str
                token to identify first sample in the scene for fetching
            output_dirpath : str
                root of output directory
            paths_only : bool
                if set, then only produce paths
    Returns:
        list[str] : paths to camera image
        list[str] : paths to lidar depth map
        list[str] : paths to radar depth map
    '''

    tag, \
        scene_id, \
        first_sample_token, \
        output_dirpath, \
        paths_only = args

    # Instantiate the first sample id
    sample_id = 0
    sample_token = first_sample_token

    camera_image_paths = []
    lidar_paths = []
    radar_points_paths = []

    print('Processing scene_id={}'.format(scene_id))

    # Iterate through all samples up to and including the last sample
    while sample_token != "":

        # Fetch a single sample
        current_sample = nusc.get('sample', sample_token)
        camera_token = current_sample['data']['CAM_FRONT']
        camera_sample = nusc.get('sample_data', camera_token)

        '''
        Set up paths
        '''
        camera_image_path = os.path.join(nusc.dataroot, camera_sample['filename'])

        dirpath, filename = os.path.split(camera_image_path)
        dirpath = dirpath.replace(nusc.dataroot, output_dirpath)
        filename = os.path.splitext(filename)[0]

        # Create lidar path
        lidar_dirpath = dirpath.replace(
            'samples',
            os.path.join('lidar', 'scene_{}'.format(scene_id)))
        lidar_filename = filename + '.png'

        lidar_path = os.path.join(
            lidar_dirpath,
            lidar_filename)

        # Create radar path
        radar_points_dirpath = dirpath.replace(
            'samples',
            os.path.join('radar_points', 'scene_{}'.format(scene_id)))
        radar_points_filename = filename + '.npy'

        radar_points_path = os.path.join(
            radar_points_dirpath,
            radar_points_filename)



        # In case multiple threads create same directory
        dirpaths = [
            lidar_dirpath,
            radar_points_dirpath
        ]

        for dirpath in dirpaths:
            if not os.path.exists(dirpath):
                try:
                    os.makedirs(dirpath)
                except Exception:
                    pass

        '''
        Store file paths
        '''
        camera_image_paths.append(camera_image_path)
        radar_points_paths.append(radar_points_path)
        lidar_paths.append(lidar_path)

        if not paths_only:

            '''
            Get camera data
            '''
            camera_image = data_utils.load_image(camera_image_path)

            '''
            Get lidar points projected to an image and save as PNG
            '''
            lidar_depth = lidar_depth_map_from_token(
                nusc=nusc,
                nusc_explorer=nusc_explorer,
                current_sample_token=sample_token)

            data_utils.save_depth(lidar_depth, lidar_path)

            '''
            Merge forward and backward point clouds for radar and lidar
            '''

            points_radar, depth_radar = get_radar_points(
                nusc=nusc,
                nusc_explorer=nusc_explorer,
                current_sample_token=sample_token)



            '''
            Save radar points as a numpy array
            '''
            radar_points = np.stack([
                points_radar[0, :],
                points_radar[1, :],
                depth_radar],
                axis=-1)

            np.save(radar_points_path, radar_points)

        '''
        Move to next sample in scene
        '''
        sample_id = sample_id + 1
        sample_token = current_sample['next']
        
        # Stop if we've reached the end of the scene
        if sample_token == "":
            break

    print('Finished {} samples in scene_id={}'.format(sample_id, scene_id))

    return (tag,
            camera_image_paths,
            lidar_paths,
            radar_points_paths)


'''
Main function
'''
if __name__ == '__main__':

    use_multithread = args.n_thread > 1 and not args.debug

    pool_inputs = []
    pool_results = []

    test_camera_image_paths = []
    test_lidar_paths = []
    test_radar_points_paths = []

    n_scenes_to_process = min(args.n_scenes_to_process, MAX_SCENES)
    print('Total Scenes to process: {}'.format(n_scenes_to_process))

    # Add all tasks for processing each scene to pool inputs
    for scene_id in range(0, min(args.n_scenes_to_process, MAX_SCENES)):

        tag = 'test'

        current_scene = nusc.scene[scene_id]
        first_sample_token = current_scene['first_sample_token']
        last_sample_token = current_scene['last_sample_token']

        inputs = [
            tag,
            scene_id,
            first_sample_token,
            args.nuscenes_data_derived_dirpath,
            args.paths_only
        ]

        pool_inputs.append(inputs)

        if not use_multithread:
            pool_results.append(process_scene(inputs))

    if use_multithread:
        # Create pool of threads
        with mp.Pool(args.n_thread) as pool:
            # Will fork n_thread to process scene
            pool_results = pool.map(process_scene, pool_inputs)

    # Unpack output paths
    for results in pool_results:

        tag, \
            camera_image_scene_paths, \
            lidar_scene_paths, \
            radar_points_scene_paths = results


        test_camera_image_paths.extend(camera_image_scene_paths)
        test_lidar_paths.extend(lidar_scene_paths)
        test_radar_points_paths.extend(radar_points_scene_paths)

    '''
    Write paths to file
    '''
    # === 수정 시작: outputs 리스트를 test set에 맞게 수정 ===
    outputs = [
        [
            'training',
            [
                [
                    'image',
                    test_camera_image_paths,
                    TEST_IMAGE_FILEPATH
                ], [
                    'lidar',
                    test_lidar_paths,
                    TEST_LIDAR_FILEPATH
                ], [
                    'radar',
                    test_radar_points_paths,
                    TEST_RADAR_FILEPATH
                ]
            ]
        ],
    ]
    # === 수정 끝 ===

    # Create output directories
    for dirpath in [TEST_REF_DIRPATH]:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    for output_info in outputs:

        tag, output = output_info
        for output_type, paths, filepath in output:

            print('Storing {} {} {} file paths into: {}'.format(
                len(paths), tag, output_type, filepath))
            data_utils.write_paths(filepath, paths)
