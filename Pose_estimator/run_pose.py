import numpy as np
import os
import cv2
import json
from tqdm import tqdm
from PIL import Image
import torch
from pathlib import Path
import glob
from collections import defaultdict

# Assuming FastPoseEstimator is defined in a separate file or earlier in the codebase
from vit_pose import FastPoseEstimator

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def find_warehouse_videos(base_path, output_base_path):
    """Find all warehouse video files and their corresponding tracklets"""
    warehouses = {}
    warehouse_pattern = os.path.join(base_path, "Warehouse_*")
    warehouse_dirs = glob.glob(warehouse_pattern)
    
    for warehouse_dir in warehouse_dirs:
        warehouse_id = os.path.basename(warehouse_dir)
        video_dir = os.path.join(warehouse_dir, "videos")
        tracklets_path = os.path.join(output_base_path, warehouse_id, 'tracklets.json')
        
        if os.path.exists(video_dir) and os.path.exists(tracklets_path):
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov']:
                video_files.extend(glob.glob(os.path.join(video_dir, ext)))
            if video_files:
                warehouses[warehouse_id] = {
                    'videos': sorted(video_files),
                    'tracklets_path': tracklets_path,
                    'output_dir': os.path.join(output_base_path, warehouse_id)
                }
    
    return warehouses

def extract_camera_id(video_path):
    """Extract camera ID from video filename"""
    filename = os.path.basename(video_path)
    if filename == "Camera.mp4":
        return "Camera"
    elif filename.startswith("Camera_"):
        return filename.replace(".mp4", "")
    else:
        return filename.replace(".mp4", "")

def process_single_video_pose(video_path, tracklets, pose_model, screen_width, screen_height):
    """Process a single video to update pose information for tracklets"""
    camera_id = extract_camera_id(video_path)
    print(f"Processing video for pose update: {video_path}")
    
    # Load video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return tracklets
    
    # Filter tracklets for this camera
    camera_tracklets = [t for t in tracklets if t['camera_id'] == camera_id]
    if not camera_tracklets:
        print(f"No tracklets found for camera {camera_id}")
        video.release()
        return tracklets
    
    # Group tracklets by frame_id
    frame_tracklets = defaultdict(list)
    for tracklet in camera_tracklets:
        frame_tracklets[tracklet['frame_id']].append(tracklet)
    
    # Process frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0
    updated_tracklets = [t for t in tracklets if t['camera_id'] != camera_id]  # Keep tracklets from other cameras
    
    with tqdm(total=total_frames, desc=f"Processing poses for {camera_id}") as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            if frame_id in frame_tracklets:
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Prepare bounding boxes
                current_tracklets = frame_tracklets[frame_id]
                bboxes = []
                for tracklet in current_tracklets:
                    bbox = tracklet['bbox_2d']
                    bboxes.append([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']])
                
                # Estimate poses
                bboxes = np.array(bboxes, dtype=np.float32)
                pose_results = pose_model.estimate_poses_batch(frame_rgb, [bboxes])[0] if bboxes.size > 0 else []
                
                # Update tracklets with pose information
                for tracklet, pose in zip(current_tracklets, pose_results):
                    keypoints_dict = {}
                    trajectory_center = None
                    keypoints = pose.get('keypoints', [])
                    scores = pose.get('scores', [])
                    
                    for k, (kp, score) in enumerate(zip(keypoints, scores)):
                        if k < len(pose_model.keypoint_names):
                            keypoints_dict[pose_model.keypoint_names[k]] = {
                                'x': float(kp[0].cpu().numpy()),
                                'y': float(kp[1].cpu().numpy()),
                                'score': float(score.cpu().numpy())
                            }
                    
                    # Calculate trajectory center
                    bbox = tracklet['bbox_2d']
                    bbox_array = np.array([bbox['x1'], bbox['y1'], bbox['x2'] - bbox['x1'], bbox['y2'] - bbox['y1']])
                    trajectory_center = pose_model.get_trajectory_center(keypoints, scores, bbox_array)
                    
                    # Update tracklet
                    tracklet['keypoints'] = keypoints_dict
                    tracklet['trajectory_center'] = [float(coord) for coord in trajectory_center] if trajectory_center is not None else None
                    updated_tracklets.append(tracklet)
            else:
                # Add unchanged tracklets for this frame
                updated_tracklets.extend(frame_tracklets.get(frame_id, []))
            
            frame_id += 1
            pbar.update(1)
    
    video.release()
    torch.cuda.empty_cache()
    return updated_tracklets

def main():
    # Hardcoded paths
    base_path = '/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/data_aic2025/raw/test'
    output_base_path = '/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/aic2025/warehouse_tracklets'
    
    # Initialize pose model
    pose_model = FastPoseEstimator(device="cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Find warehouse videos and tracklets
    print("Scanning for warehouse videos and tracklets...")
    warehouses = find_warehouse_videos(base_path, output_base_path)
    
    if not warehouses:
        print("No warehouses with tracklets found!")
        return
    
    print(f"Found {len(warehouses)} warehouses to process:")
    for warehouse_id, data in warehouses.items():
        print(f"  {warehouse_id}: {len(data['videos'])} videos")
    
    # Process each warehouse
    for warehouse_id, data in warehouses.items():
        print(f"\n{'='*50}")
        print(f"Processing {warehouse_id}")
        print(f"{'='*50}")
        
        # Load existing tracklets
        tracklets_path = data['tracklets_path']
        if not os.path.exists(tracklets_path):
            print(f"Tracklets file not found for {warehouse_id}")
            continue
        
        with open(tracklets_path, 'r') as f:
            tracklets = json.load(f)
        
        print(f"Loaded {len(tracklets)} tracklets from {tracklets_path}")
        
        # Process each video
        updated_tracklets = tracklets
        for video_path in data['videos']:
            try:
                # Get video dimensions
                video = cv2.VideoCapture(video_path)
                screen_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                screen_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video.release()
                
                updated_tracklets = process_single_video_pose(
                    video_path, updated_tracklets, pose_model, screen_width, screen_height
                )
                
                print(f"Updated poses for {os.path.basename(video_path)}")
                
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
        
        # Save updated tracklets
        output_path = os.path.join(data['output_dir'], 'tracklets.json')
        with open(output_path, 'w') as f:
            json.dump(updated_tracklets, f, indent=2, cls=NumpyEncoder)
        
        print(f"Saved updated tracklets to {output_path}")
        print(f"Total tracklets: {len(updated_tracklets)}")
    
    print(f"\nProcessing complete! Updated results saved to: {output_base_path}")

if __name__ == '__main__':
    main()
