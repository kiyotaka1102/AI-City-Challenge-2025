import numpy as np
import json
import os
import glob
import argparse
from functools import partial
from scipy.optimize import linear_sum_assignment
from util.camera import Camera
from Tracker.kalman_filter_box_zya import KalmanFilter_box
from Tracker.PoseTracker import PoseTracker, Detection_Sample
from Solver.bip_solver import GLPKSolver
import aic_cpp
from tqdm import tqdm
import h5py
from util.func_3d_bbox import *
import math
import gc
def compute_3d_params(detection, depth_frame, camera):
    """
    Compute width, height, length, and yaw3d for a Detection_Sample using depth data.
    
    Args:
        detection: Detection_Sample object containing bbox, keypoints_2d, and class_id.
        depth_frame: Numpy array of depth values (in millimeters).
        camera: Camera object with extrinsic_mat and intrinsic_mat.
    
    Returns:
        tuple: (w_final, h_final, l_final, yaw_rad, world_bbox)
    """
    if depth_frame is None:
        return None, None, None, None, None

    # Compute bbox center
    x1, y1, x2, y2, _ = detection.bbox
    bbox_center = get_center_obj(detection)
    

    # Extract keypoints in the format expected by compute_wl and compute_yaw
    keypoints_map = {
        'Nose': 0, 'L_Eye': 1, 'R_Eye': 2, 'L_Ear': 3, 'R_Ear': 4,
        'L_Shoulder': 5, 'R_Shoulder': 6, 'L_Hip': 11, 'R_Hip': 12,
        'L_Ankle': 15, 'R_Ankle': 16
    }
    pts = {}
    for kp_name, idx in keypoints_map.items():
        x, y, score = detection.keypoints_2d[idx]
        pts[kp_name] = (x, y) if score > 0 else None

    # Get camera parameters
    E = camera.extrinsic_mat
    K = camera.intrinsic_mat
    camera_intrinsics = {
        'fx': K[0, 0], 'fy': K[1, 1],
        'cx': K[0, 2], 'cy': K[1, 2]
    }
    scaleFactor = camera.scale_factor
    x_origin = camera.translation_to_global[0]
    y_origin = camera.translation_to_global[1]
    Homo = camera.homo_mat

    # Compute height
    h_pixel = y2 - y1  # Height of bounding box in pixels
    h_final = compute_h(h_pixel, bbox_center, depth_frame, E, K)

    # Compute width and length
    w_final, l_final = compute_wl(pts, detection.bbox, detection.class_id, depth_frame, E, K)
    yaw_rad, world_bbox = compute_yaw(pts, depth_frame, E, K, bbox_center)

    # Compute yaw
    if (detection.class_id not in [0] ) and detection.yaw2d is not None:  # Not Person
        # print(detection.yaw2d)
        if (detection.class_id in [4,5] and detection.keypoints_2d[5][-1]<0.3 and detection.keypoints_2d[6][-1] <0.3 ) or detection.class_id not in [5,4]:
            yaw_rad = compute_yaw3d_from_yaw2d(bbox_center, detection.yaw2d, depth_frame, E, K)
    # # Compute yaw based on class_id
    # if detection.class_id == 5: 
    #     # Create mask for white humanoid
    #     cropped_depth = depth_frame[int(y1):int(y2), int(x1):int(x2)]
    #     if cropped_depth.size == 0:
    #         return w_final, h_final, l_final, yaw_rad, world_bbox

    #     H, W = cropped_depth.shape
    #     patch_height = H // 10
    #     mask = np.zeros_like(cropped_depth, dtype=np.uint8)
    #     for i in range(10):
    #         y_start = i * patch_height
    #         y_end = H if i == 9 else (i + 1) * patch_height
    #         patch = cropped_depth[y_start:y_end, :]
    #         if patch.size == 0:
    #             continue
    #         bottom_row = patch[-1, :]
    #         if bottom_row.size > 0:
    #             patch_thresh = np.max(bottom_row)
    #             patch_mask = (patch > 0) & (patch < patch_thresh)
    #             mask[y_start:y_end, :] = patch_mask.astype(np.uint8)
    #     mask *= 255

    #     result = calculate_3d_yaw_white_humanoid(mask, cropped_depth, camera_intrinsics, x1, y1)
    #     if result is not None and result[3] is not None:
    #         start_point_2d_full, end_point_2d_full = result[2], result[3]
    #         start_bev = to_bev(start_point_2d_full, Homo, x_origin, y_origin, scaleFactor)
    #         end_bev = to_bev(end_point_2d_full, Homo, x_origin, y_origin, scaleFactor)
    #         dx = end_bev[0] - start_bev[0]
    #         dy = end_bev[1] - start_bev[1]
    #         yaw_rad = np.arctan2(dx, dy)

    # if detection.class_id == 2:  # Novacarter
    #     cropped_depth = depth_frame[int(y1-10):int(y2+15), int(x1-10):int(x2+15)]
    #     if cropped_depth.size == 0:
    #         return w_final, h_final, l_final, yaw_rad, world_bbox

    #     offset_x, offset_y = x1-10, y1-10
    #     valid_pixels = cropped_depth[cropped_depth > 0]
    #     if valid_pixels.size == 0:
    #         return w_final, h_final, l_final, yaw_rad, world_bbox
        
    #     thresh = np.percentile(valid_pixels, 60)
    #     mask = (cropped_depth > 0) & (cropped_depth < thresh)
    #     mask = mask.astype(np.uint8) * 255
        
    #     kernel = np.ones((3, 3), np.uint8)
    #     mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    #     mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    #     contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #     if not contours:
    #         return w_final, h_final, l_final, yaw_rad, world_bbox
    #     largest = max(contours, key=cv.contourArea)
    #     mask_clean = np.zeros_like(mask)
    #     cv.drawContours(mask_clean, [largest], -1, 255, thickness=cv.FILLED)

    #     edges = cv.Canny(mask_clean, 1, 1)
    #     lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=25, minLineLength=20, maxLineGap=5)
    #     lines_info = []

    #     h, w = mask_clean.shape
    #     margin = 2
    #     if lines is not None:
    #         for line in lines:
    #             lx1, ly1, lx2, ly2 = line[0]
    #             if is_near_border(lx1, ly1, w, h, margin) or is_near_border(lx2, ly2, w, h, margin):
    #                 continue
    #             dx, dy = lx2 - lx1, ly2 - ly1
    #             angle = np.arctan2(dy, dx)
    #             length = np.hypot(dx, dy)
    #             lines_info.append((angle, length, (lx1, ly1, lx2, ly2)))

    #     if not lines_info:
    #         return w_final, h_final, l_final, yaw_rad, world_bbox
        
    #     if len(lines_info) == 1:
    #         longest_line = lines_info[0][2]
    #     else:
    #         lengths = [line[1] for line in lines_info]
    #         length_thresh = np.percentile(lengths, 80)
    #         long_lines = [line for line in lines_info if line[1] >= length_thresh]
    #         best_group = find_best_angle_group(long_lines)
    #         if not best_group:
    #             best_group = find_best_angle_group(lines_info)
    #         longest_line = max(best_group, key=lambda x: x[1])[2] if best_group else None

    #     if longest_line is not None:
    #         M = cv.moments(mask)
    #         if M["m00"] > 0:
    #             cx = int(M["m10"] / M["m00"])
    #             cy = int(M["m01"] / M["m00"])
    #             lx1, ly1, lx2, ly2 = longest_line
    #             vx, vy = lx2 - lx1, ly2 - ly1
    #             norm = np.hypot(vx, vy)
    #             if norm > 0:
    #                 vx /= norm
    #                 vy /= norm
    #                 length = 60
    #                 end = (int(cx + vx * length), int(cy + vy * length))
    #                 start_full = (int(cx + offset_x), int(cy + offset_y))
    #                 end_full = (int(end[0] + offset_x), int(end[1] + offset_y))
    #                 start_bev = to_bev(start_full, Homo, x_origin, y_origin, scaleFactor)
    #                 end_bev = to_bev(end_full, Homo, x_origin, y_origin, scaleFactor)
    #                 dx = end_bev[0] - start_bev[0]
    #                 dy = end_bev[1] - start_bev[1]
    #                 yaw_rad = np.arctan2(dx, dy) 
    
    
    if detection.class_id ==1:
        adjusted_dimension = np.array([min(w_final,2.6),min(l_final,1.8),min(h_final,2.3)])
    elif detection.class_id ==2:
        yaw_rad+=np.pi/2
        adjusted_dimension = np.array([min(w_final, 0.75), min(l_final,0.47), min(h_final, 0.37)])
        # world_bbox[0],world_bbox[1] = to_bev(bbox_center,Homo,x_origin,y_origin,scaleFactor)
        # center_3d[2]= adjusted_dimension[2]/2 
    elif detection.class_id ==3:
        yaw_rad+=np.pi/2
        adjusted_dimension = np.array([min(w_final, 1.32),min(l_final,0.65), min(h_final, 0.2)])
        # world_bbox[0],world_bbox[1]  = to_bev(bbox_center,Homo,x_origin,y_origin,scaleFactor)
        # center_3d[2]= adjusted_dimension[2]/2 
    elif detection.class_id ==5:
        adjusted_dimension = np.array([0.53,0.85,1.8])
        # world_bbox[2]= adjusted_dimension[2]/2 
    else:
        adjusted_dimension = np.array([min(max(w_final,0.3), 0.6), min(l_final,0.85), min(max(h_final,1.6), 1.8)])
        # world_bbox[2]= adjusted_dimension[2]/2 
    world_bbox[2]= adjusted_dimension[2]/2 
    return adjusted_dimension[0], adjusted_dimension[2], adjusted_dimension[1], yaw_rad, world_bbox

def load_depth_map(depth_dir, cam_id, frame_id):
    """
    Load depth map for a specific camera and frame from an HDF5 file.
    """
    # Map cam_id to file name
    if cam_id == "Camera":
        # Handle the case where cam_id is just "Camera"
        cam_file = "Camera"
    elif cam_id.startswith("Camera_"):
        # Already in correct format (e.g., "Camera_01")
        cam_file = cam_id
    else:
        # Handle numeric IDs (e.g., "01" -> "Camera_01")
        cam_file = f"Camera_{cam_id.zfill(2)}"
    
    depth_file = os.path.join(depth_dir, f"{cam_file}.h5")
    
    if not os.path.exists(depth_file):
        print(f"Depth file {depth_file} not found")
        return None
    
    try:
        with h5py.File(depth_file, 'r') as h5f:
            keys = list(h5f.keys())
            if frame_id < len(keys):
                depth_frame = h5f[keys[frame_id]][()].astype(np.float16)
                return depth_frame
            else:
                print(f"Frame {frame_id} not found in {depth_file}")
                return None
    except Exception as e:
        print(f"Error loading depth map from {depth_file}: {e}")
        return None
    

def load_cameras(calibration_path):
    """
    Load camera configurations from a calibration JSON file using the provided Camera class.
    Returns a list of Camera objects with populated attributes.
    """
    if not os.path.exists(calibration_path):
        print(f"Error: Calibration file not found at {calibration_path}")
        return []

    try:
        with open(calibration_path, 'r') as f:
            calib_data = json.load(f)
        
        cameras = []
        
        # Check if calibration data contains a 'sensors' array
        if 'sensors' in calib_data and isinstance(calib_data['sensors'], list):
            for sensor in calib_data['sensors']:
                if sensor.get('type') != 'camera':
                    continue
                
            # Handle translationToGlobalCoordinates
                translation = sensor.get('translationToGlobalCoordinates', {})
                try:
                    x = float(translation.get('x', 0.0))
                    y = float(translation.get('y', 0.0))
                    z = float(translation.get('z', 0.0))  # Default to 0.0 if z is missing
                    translation_list = [x, y, z]
                except (ValueError, TypeError) as e:
                    print(f"Invalid translationToGlobalCoordinates in sensor {sensor.get('id')}: {translation}, error={e}")
                    continue
                
                cam_data = {
                    'cameraMatrix': sensor.get('cameraMatrix'),
                    'homography': sensor.get('homography'),
                    'id': sensor.get('id', 'Camera'),
                    'intrinsicMatrix': sensor.get('intrinsicMatrix'),
                    'extrinsicMatrix': sensor.get('extrinsicMatrix'),
                    'scaleFactor': sensor.get('scaleFactor', 1.0),
                    'translationToGlobalCoordinates': translation_list
                }
                
                # Create Camera object with sensor_id
                cam = Camera(cam_data)
                cameras.append(cam)

        
        cameras.sort(key=lambda x: x.idx_int)
        print(f"Loaded {len(cameras)} cameras from {calibration_path}:")
        for cam in cameras:
            print(f"  Camera ID: {cam.idx}, idx_int: {cam.idx_int}, full_id: {cam.full_id}")
        return cameras
    
    except Exception as e:
        print(f"Error loading calibration file {calibration_path}: {e}")
        return []

def load_tracklets_and_features(tracklet_dir, valid_class_ids=[0,1,2,3,4,5]):
    """
    Load tracklets and features from a tracklet directory.
    Filters tracklets by valid_class_ids (e.g., [0] for persons).
    Returns a dictionary mapping camera_id to tracklet data and features.
    """
    tracklets_path = os.path.join(tracklet_dir, 'tracklets.json')
    features_path = os.path.join(tracklet_dir, 'features.npy')
    feature_mapping_path = os.path.join(tracklet_dir, 'feature_mapping.json')

    if not os.path.exists(tracklets_path) or not os.path.exists(features_path):
        print(f"Error: Missing tracklets or features in {tracklet_dir}")
        return None, None

    with open(tracklets_path, 'r') as f:
        tracklets = json.load(f)

    features = np.load(features_path)
    with open(feature_mapping_path, 'r') as f:
        feature_mapping = json.load(f)

    tracklets_by_cam = {}
    for tracklet in tracklets:
        cam_id = tracklet['camera_id']
        frame_id = tracklet['frame_id']
        class_id = tracklet.get('class_id', -1)
        
        if class_id not in valid_class_ids:
            continue
            
        if cam_id not in tracklets_by_cam:
            tracklets_by_cam[cam_id] = {}
        if frame_id not in tracklets_by_cam[cam_id]:
            tracklets_by_cam[cam_id][frame_id] = []
        tracklets_by_cam[cam_id][frame_id].append(tracklet)

    features_by_cam = {}
    for i, tracklet in enumerate(tracklets):
        cam_id = tracklet['camera_id']
        frame_id = tracklet['frame_id']
        class_id = tracklet.get('class_id', -1)
        feature_idx = tracklet.get('feature_idx', -1)
        
        if class_id not in valid_class_ids or feature_idx < 0 or feature_idx >= len(features):
            continue
            
        if cam_id not in features_by_cam:
            features_by_cam[cam_id] = {}
        if frame_id not in features_by_cam[cam_id]:
            features_by_cam[cam_id][frame_id] = []
        features_by_cam[cam_id][frame_id].append(features[feature_idx])

    print(f"Loaded tracklets for cameras: {list(tracklets_by_cam.keys())}")
    return tracklets_by_cam, features_by_cam

def create_detection_samples(tracklets, features, cam_id, frame_id, cam_idx_map, cameras, depth_dir):
    """
    Create Detection_Sample objects for a specific camera and frame.
    Incorporates trajectory_center, yaw, and 3D dimensions using depth data.
    """
    samples = []
    if cam_id not in tracklets or frame_id not in tracklets[cam_id]:
        return samples

    # Find the camera object
    cam_idx = cam_idx_map.get(cam_id, -1)
    camera = next((cam for cam in cameras if cam.idx_int == cam_idx), None)
    if not camera:
        print(f"Camera {cam_id} not found in calibration")
        return samples
    # Load depth map
    depth_frame = load_depth_map(depth_dir, cam_id, frame_id)

    tracklets_frame = tracklets[cam_id][frame_id]
    features_frame = features.get(cam_id, {}).get(frame_id, [])

    for i, tracklet in enumerate(tracklets_frame):
        bbox = np.array([
            tracklet['bbox_2d']['x1'],
            tracklet['bbox_2d']['y1'],
            tracklet['bbox_2d']['x2'],
            tracklet['bbox_2d']['y2'],
            tracklet['bbox_2d']['confidence']
        ], dtype=np.float16)
        keypoints_2d = np.zeros((17, 3) , dtype=np.float16)

        # Populate keypoints
        for kp_name, kp_data in tracklet['keypoints'].items():
            idx = {
                'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
                'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
                'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
                'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
            }.get(kp_name)
            if idx is not None:
                keypoints_2d[idx] = [kp_data['x'], kp_data['y'], kp_data['score']]
            else: 
                print('Warning have unknown pose')


        reid_feat = features_frame[i] if i < len(features_frame) else np.zeros(2048)
        reid_feat = reid_feat.astype(np.float16)
        if np.any(reid_feat != 0):  # Normalize only if not a zero vector
            reid_feat = reid_feat / np.linalg.norm(reid_feat)
        class_id = tracklet.get('class_id', -1)
        # print(class_id)
        obj_id = tracklet.get('obj_id', -1)
        trajectory_center = np.array(tracklet['trajectory_center']) if tracklet.get('trajectory_center') else None
        yaw = tracklet.get('yaw', None)

        # Compute 3D parameters
        w, h, l, yaw3d, world_bbox = compute_3d_params(
            Detection_Sample(bbox, keypoints_2d, reid_feat, cam_idx, frame_id, class_id, yaw=yaw['yaw_angle']), 
            depth_frame, 
            camera
        )

        
        sample = Detection_Sample(
            bbox=bbox,
            keypoints_2d=keypoints_2d,
            reid_feat=reid_feat,
            cam_id=cam_idx,
            frame_id=frame_id,
            class_id=class_id,
            obj_id=obj_id,
            trajectory_center=trajectory_center,
            yaw=yaw['yaw_angle'],
            width=w,
            length=l,
            height=h,
            yaw3d=yaw3d,
            bbox_center3d = world_bbox,
            # depth_frame=depth_frame
        )
        samples.append(sample)

    return samples

def generate_global_tracks(warehouse_dir, tracklet_dir, output_dir, max_frames=1900):
    """
    Generate global tracks from tracklets for a given warehouse, including 3D dimensions and yaw.
    """
    calibration_path = os.path.join(warehouse_dir, 'calibration.json')
    depth_dir = os.path.join(warehouse_dir, 'depth_maps')
    cameras = load_cameras(calibration_path)
    if not cameras:
        print(f"No cameras loaded for {warehouse_dir}")
        return

    cam_idx_map = {}
    for cam in cameras:
        cam_idx_map[cam.idx] = cam.idx_int
        cam_idx_map[cam.full_id] = cam.idx_int
        if cam.full_id.startswith("Camera_"):
            short_id = cam.full_id.replace("Camera_", "")
            cam_idx_map[short_id] = cam.idx_int
    
    print(f"Camera index map: {cam_idx_map}")

    tracklets_by_cam, features_by_cam = load_tracklets_and_features(tracklet_dir, valid_class_ids=[0,1,2,3,4,5])
    if not tracklets_by_cam:
        print(f"No tracklets found for {tracklet_dir}")
        return

    tracker = PoseTracker(cameras)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'global_tracks.txt')

    with open(output_path, 'w') as f:
        for frame_id in tqdm(range(max_frames), desc="Processing frames"):
            detection_sample_list_mv = [[] for _ in range(len(cameras))]
            for cam_id in tracklets_by_cam.keys():
                if cam_id not in cam_idx_map:
                    continue
                cam_idx = cam_idx_map[cam_id]
                samples = create_detection_samples(
                    tracklets_by_cam, features_by_cam, cam_id, frame_id, cam_idx_map, cameras, depth_dir
                )
                detection_sample_list_mv[cam_idx] = samples

            tracker.mv_update_wo_pred(detection_sample_list_mv, frame_id)
            frame_results = tracker.output(frame_id)
            
            if frame_results:
                frame_results = np.concatenate(frame_results, axis=0)
                sort_idx = np.lexsort((frame_results[:,2], frame_results[:,0]))
                frame_results = frame_results[sort_idx]
                np.savetxt(
                    f,
                    frame_results,
                    fmt='%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f'
                )
            detection_sample_list_mv = None
            gc.collect()
        print(f"Global tracks saved to {output_path}")
    # else:
    #     print(f"No results to save for {warehouse_dir}")

    print(f"Global tracks saved to {output_path}")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, 
                        default='/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/data_aic2025/raw/test',
                        help='Base path containing warehouse directories with calibration files')
    parser.add_argument('--tracklet_base_path', type=str, 
                        default='/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/aic2025/warehouse_tracklets',
                        help='Base path containing warehouse directories with tracklet files')
    parser.add_argument('--output_dir', type=str, default='global_tracks_test',
                        help='Output directory for global tracks')
    parser.add_argument('--warehouses', type=str, nargs='*',
                        help='Specific warehouses to process (e.g., Warehouse_018)')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    warehouse_pattern = os.path.join(args.base_path, "Warehouse_*")
    warehouse_dirs = glob.glob(warehouse_pattern)

    if not warehouse_dirs:
        print("No warehouse directories found!")
        return

    if args.warehouses:
        warehouse_dirs = [d for d in warehouse_dirs if os.path.basename(d) in args.warehouses]

    print(f"Found {len(warehouse_dirs)} warehouses to process:")
    for warehouse_dir in warehouse_dirs:
        warehouse_id = os.path.basename(warehouse_dir)
        print(f"  {warehouse_id}")
        if warehouse_id not in  ['Warehouse_018','Warehouse_019']:
            continue
        tracklet_dir = os.path.join(args.tracklet_base_path, warehouse_id)
        warehouse_output_dir = os.path.join(args.output_dir, warehouse_id)
        generate_global_tracks(warehouse_dir, tracklet_dir, warehouse_output_dir, max_frames=9000)

    print(f"\nProcessing complete! Global tracks saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
