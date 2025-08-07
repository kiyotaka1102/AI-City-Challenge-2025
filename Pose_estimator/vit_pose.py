import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import defaultdict
# from IPython.display import clear_output
import os
import json
from scipy.optimize import linear_sum_assignment
import torch
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)
from PIL import Image
# import pickleclass

class FastPoseEstimator:
    """Optimized pose estimation with batching and caching"""
    
    def __init__(self, device="cuda:1" if torch.cuda.is_available() else "cpu", batch_size=8):
        self.device = device
        self.batch_size = batch_size
        print(f"🤖 Initializing pose estimation models on {device}...")
        
        # Initialize pose estimation model
        self.pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-base")
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(
            "usyd-community/vitpose-plus-base", 
            device_map=device
        )
        
        # Define COCO 17-keypoint names and their indices
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        self.keypoint_index_map = {name: idx for idx, name in enumerate(self.keypoint_names)}
        
        # Define ankle keypoints for trajectory
        self.ankle_keypoints = ['left_ankle', 'right_ankle']
        
        # Minimum number of keypoints to consider a detection valid
        self.min_keypoints = 1
        self.min_avg_score = 0.0
        
        print("✅ Pose estimation models loaded successfully!")
    
    def estimate_poses_batch(self, image, boxes_list):
        """Estimate poses for multiple detections efficiently without filtering keypoints"""
        if not boxes_list or all(len(boxes) == 0 for boxes in boxes_list):
            return [[] for _ in boxes_list]
        
        try:
            # Convert PIL Image to numpy if needed
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image
            
            all_results = []
            
            # Process each detection set
            for boxes in boxes_list:
                if len(boxes) == 0:
                    all_results.append([])
                    continue
                
                # Convert boxes to expected format [x, y, w, h]
                formatted_boxes = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    formatted_boxes.append([x1, y1, w, h])
                
                # Process in smaller batches to avoid memory issues
                pose_results = []
                for i in range(0, len(formatted_boxes), self.batch_size):
                    batch_boxes = formatted_boxes[i:i + self.batch_size]
                    
                    # Process image for pose estimation
                    inputs = self.pose_processor(
                        image_np, 
                        boxes=[batch_boxes], 
                        return_tensors="pt"
                    ).to(self.device)
                    inputs["dataset_index"] = torch.tensor([0], device=self.device)
                    
                    with torch.no_grad():
                        outputs = self.pose_model(**inputs)
                    
                    # Post-process results
                    batch_results = self.pose_processor.post_process_pose_estimation(
                        outputs, 
                        boxes=[batch_boxes], 
                    )
                    
                    if batch_results and len(batch_results) > 0:
                        pose_results.extend(batch_results[0])
                
                # Include all results without filtering
                all_results.append(pose_results)
            
            return all_results
            
        except Exception as e:
            print(f"⚠️ Error in batch pose estimation: {e}")
            return [[] for _ in boxes_list]
    def get_trajectory_center(self, keypoints, scores, bbox=None):
        """Calculate trajectory center using ankle keypoints or bounding box."""
        num_keypoints = min(len(keypoints), len(scores))
        if num_keypoints < self.min_keypoints or bbox is None:
            if isinstance(bbox, dict):
                return np.array([float(bbox['xc']), float(bbox['yc']) + float(bbox['h']) / 2])
            elif isinstance(bbox, np.ndarray):
                x1, y1, w, h = bbox
                return np.array([x1 + w / 2, y1 + h])
            return None

        # Extract bounding box parameters
        if isinstance(bbox, dict):
            x1 = float(bbox['xc']) - float(bbox['w']) / 2
            y1 = float(bbox['yc']) - float(bbox['h']) / 2
            w, h = float(bbox['w']), float(bbox['h'])
            xc = float(bbox['xc'])
        elif isinstance(bbox, np.ndarray):
            x1, y1, w, h = bbox
            xc = x1 + w / 2
        else:
            return None

        # Convert keypoints and scores to dictionaries
        keypoint_dict = {self.keypoint_names[i]: keypoints[i] for i in range(min(num_keypoints, len(self.keypoint_names)))}
        score_dict = {self.keypoint_names[i]: scores[i] for i in range(min(num_keypoints, len(self.keypoint_names)))}

        # Collect valid ankle keypoints (score > 0.3)
        valid_ankles = []
        for kp_name in self.ankle_keypoints:
            if kp_name in keypoint_dict and score_dict[kp_name] > 0.3:
                valid_ankles.append(keypoint_dict[kp_name].cpu().numpy())

        # Calculate trajectory center
        if len(valid_ankles) == 2:
            trajectory_center = np.mean(valid_ankles, axis=0)
        elif len(valid_ankles) == 1:
            trajectory_center = np.array([xc, valid_ankles[0][1]])
        else:
            return np.array([xc, y1 + h])

        # Constrain to bounding box
        x2, y2 = x1 + w, y1 + h
        trajectory_center[0] = np.clip(trajectory_center[0], x1, x2)
        trajectory_center[1] = np.clip(trajectory_center[1], y1, y2)

        # Validate distance from bbox bottom center
        bbox_bottom_center = np.array([xc, y1 + h])
        max_offset = max(w / 2, h / 2)
        if np.linalg.norm(trajectory_center - bbox_bottom_center) > max_offset:
            return bbox_bottom_center

        return trajectory_center

