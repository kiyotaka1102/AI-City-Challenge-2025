import numpy as np
import os
import torch
import torch.nn.functional as F
import cv2
import sys
import json
import pickle
from tqdm import tqdm
from torchvision.ops import roi_align
from ultralytics import YOLO
import argparse
from collections import defaultdict, deque
from pathlib import Path
import glob
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor, VitPoseForPoseEstimation
from huggingface_hub import hf_hub_download
from scipy.optimize import linear_sum_assignment 
from OrientAnything.vision_tower import DINOv2_MLP
import json
import time
from concurrent.futures import ThreadPoolExecutor
import threading


# Add fast-reid to path (adjust as needed)
current_file_path = os.path.abspath(__file__)
path_arr = current_file_path.split('/')[:-1]
root_path = '/'.join(path_arr)
print(root_path)
sys.path.append(os.path.join(root_path,'fast-reid'))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class OptimizedReidInferencer():
    def __init__(self, reid, batch_size=32):
        self.reid = reid
        self.batch_size = batch_size
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(reid.device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(reid.device)
        self.device = self.reid.device
        
        self.tensor_cache = {}
        for bs in [1, 2, 4, 8, 16, 32]:
            self.tensor_cache[bs] = torch.zeros(bs, 1, dtype=torch.float32, device=self.device)
    
    def mgn_batch(self, crops):
        """Optimized batch processing for MGN features"""
        with torch.no_grad():
            features = self.reid.backbone(crops)
            b1_feat = self.reid.b1(features)
            b2_feat = self.reid.b2(features)
            b21_feat, b22_feat = torch.chunk(b2_feat, 2, dim=2)
            b3_feat = self.reid.b3(features)
            b31_feat, b32_feat, b33_feat = torch.chunk(b3_feat, 3, dim=2)

            b1_pool_feat = self.reid.b1_head(b1_feat)
            b2_pool_feat = self.reid.b2_head(b2_feat)
            b21_pool_feat = self.reid.b21_head(b21_feat)
            b22_pool_feat = self.reid.b22_head(b22_feat)
            b3_pool_feat = self.reid.b3_head(b3_feat)
            b31_pool_feat = self.reid.b31_head(b31_feat)
            b32_pool_feat = self.reid.b32_head(b32_feat)
            b33_pool_feat = self.reid.b33_head(b33_feat)

            pred_feat = torch.cat([b1_pool_feat, b2_pool_feat, b3_pool_feat, b21_pool_feat,
                                  b22_pool_feat, b31_pool_feat, b32_pool_feat, b33_pool_feat], dim=1)
            return pred_feat
    
    def process_frame_batch_optimized(self, frame, bboxes_list):
        """Process multiple frames in batches for better GPU utilization"""
        if not bboxes_list or all(len(bboxes) == 0 for bboxes in bboxes_list):
            return [np.array([]).reshape(0, -1) for _ in bboxes_list]
        
        # Convert frame once
        if isinstance(frame, np.ndarray):
            frame_tensor = torch.from_numpy(frame[:, :, ::-1].copy()).permute(2, 0, 1).to(self.device)
            frame_tensor = frame_tensor / 255.0
            frame_tensor.sub_(self.mean).div_(self.std)
            frame_tensor = frame_tensor.unsqueeze(0)
        else:
            frame_tensor = frame
        
        all_features = []
        
        # Process all bboxes from all frames together
        all_crops = []
        batch_sizes = []
        
        for bboxes in bboxes_list:
            if len(bboxes) == 0:
                batch_sizes.append(0)
                continue
                
            batch_sizes.append(len(bboxes))
            cbboxes = bboxes.copy().astype(np.float32)
            
            # Get batch index tensor
            batch_idx = self.tensor_cache.get(len(cbboxes))
            if batch_idx is None or batch_idx.shape[0] != len(cbboxes):
                batch_idx = torch.zeros(len(cbboxes), 1, device=self.device)
            else:
                batch_idx = batch_idx[:len(cbboxes)]
            
            crops = roi_align(frame_tensor, 
                            torch.cat([batch_idx, torch.from_numpy(cbboxes).to(self.device)], 1), 
                            (384, 128))
            all_crops.append(crops)
        
        # Process in batches
        if all_crops:
            all_crops_tensor = torch.cat(all_crops, dim=0)
            
            # Process in batches to avoid memory issues
            features_list = []
            for i in range(0, len(all_crops_tensor), self.batch_size):
                batch_crops = all_crops_tensor[i:i + self.batch_size]
                batch_features = (self.mgn_batch(batch_crops) + self.mgn_batch(batch_crops.flip(3))) / 2
                features_list.append(batch_features.cpu())
            
            all_features_tensor = torch.cat(features_list, dim=0).numpy()
            
            # Split back to original structure
            start_idx = 0
            result_features = []
            for batch_size in batch_sizes:
                if batch_size == 0:
                    result_features.append(np.array([]).reshape(0, -1))
                else:
                    result_features.append(all_features_tensor[start_idx:start_idx + batch_size])
                    start_idx += batch_size
            
            return result_features
        
        return [np.array([]).reshape(0, -1) for _ in bboxes_list]


class FastOrientationEstimator:
    def __init__(self, checkpoint_path, device='cuda:1', batch_size=16):
        self.device = device
        self.batch_size = batch_size
        self.NUM_BINS_ROT = 360
        self.MIN_IMAGE_SIZE = 10
        
        if DINOv2_MLP is None:
            print("Warning: DINOv2_MLP not available. Orientation estimation disabled.")
            self.model = None
            return
            
        # Model configuration
        MODEL_CONFIG = {
            'dino_mode': 'base',
            'in_dim': 768,
            'out_dim': self.NUM_BINS_ROT + 1,
            'evaluate': True,
            'mask_dino': False,
            'frozen_back': True
        }
        
        self.model = DINOv2_MLP(**MODEL_CONFIG)
        self.model = self.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device) if self.model else None
        if self.model:
            self.model.eval()
        
        self.preprocessor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", cache_dir='./')
    
    def load_checkpoint(self, local_checkpoint_path):
        try:
            if os.path.exists(local_checkpoint_path):
                checkpoint = torch.load(local_checkpoint_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"Loaded orientation checkpoint from {local_checkpoint_path}")
            else:
                print(f"Warning: Orientation checkpoint not found at {local_checkpoint_path}")
                return None
        except Exception as e:
            print(f"Error loading orientation checkpoint: {e}")
            return None
        return self.model
    
    def get_yaw_angles_batch(self, image_crops):
        """Process multiple crops at once for better efficiency"""
        if self.model is None or not image_crops:
            return [None] * len(image_crops)
            
        try:
            # Convert all crops to PIL Images
            pil_crops = []
            for crop in image_crops:
                if isinstance(crop, np.ndarray):
                    pil_crops.append(Image.fromarray(crop).convert('RGB'))
                else:
                    pil_crops.append(crop.convert('RGB'))
            
            # Process in batches
            results = []
            for i in range(0, len(pil_crops), self.batch_size):
                batch_crops = pil_crops[i:i + self.batch_size]
                
                # Preprocess batch
                image_inputs = self.preprocessor(images=batch_crops, return_tensors="pt")
                image_inputs['pixel_values'] = image_inputs['pixel_values'].to(self.device)
                
                with torch.no_grad():
                    dino_pred = self.model(image_inputs)
                
                rot_logits = dino_pred[:, :self.NUM_BINS_ROT]
                conf_logits = dino_pred[:, self.NUM_BINS_ROT]
                
                rot_pred = torch.argmax(rot_logits, dim=-1)
                yaw_degrees = rot_pred.float() + 1
                yaw_degrees = yaw_degrees % 360
                yaw_degrees = torch.where(yaw_degrees > 180, yaw_degrees - 360, yaw_degrees)
                yaw_angle = torch.deg2rad(yaw_degrees)
                
                confidence = torch.sigmoid(conf_logits)
                
                # Convert to list of dictionaries
                for j in range(len(batch_crops)):
                    results.append({
                        'yaw_angle': yaw_angle[j].cpu().numpy().item(),
                        'yaw_degrees': yaw_degrees[j].cpu().numpy().item(),
                        'confidence': confidence[j].cpu().numpy().item()
                    })
            
            return results
            
        except Exception as e:
            print(f"Error in batch orientation estimation: {e}")
            return [None] * len(image_crops)

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
        """Estimate poses for multiple detections efficiently"""
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
                
                # Filter results
                filtered_results = []
                for person_pose in pose_results:
                    keypoints = person_pose['keypoints']
                    scores = person_pose['scores']
                    num_keypoints = len(keypoints)
                    avg_score = float(torch.mean(scores)) if len(scores) > 0 else 0.0
                    
                    if num_keypoints >= self.min_keypoints and avg_score >= self.min_avg_score:
                        filtered_results.append(person_pose)
                
                all_results.append(filtered_results)
            
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


def yolo_detect_optimized(model, frame, conf_thresh=0.3, valid_class_ids=[0, 1, 2, 3, 4, 5]):
    """
    Perform object detection using a YOLO model with optional class filtering.

    Args:
        model: Preloaded YOLO model from ultralytics
        frame: Input image (numpy array or path)
        conf_thresh: Minimum confidence threshold for detection
        valid_class_ids: List of class IDs to keep

    Returns:
        detections: ndarray [x1, y1, x2, y2, conf, cls_id]
        track_ids: list of -1 (placeholder, no tracking)
        class_ids: list of detected class IDs
    """
    try:
        model.to('cuda:1')
        with torch.no_grad():
            results = model(frame, verbose=False, conf=conf_thresh, device=1)

        detections = []
        track_ids = []
        class_ids = []

        for result in results:
            boxes = result.boxes
            if boxes is not None and boxes.conf is not None:
                # Filter for valid class IDs
                cls = boxes.cls.cpu().numpy()
                valid_mask = np.isin(cls, valid_class_ids)

                person_boxes = boxes.xyxy[valid_mask].cpu().numpy()
                person_confs = boxes.conf[valid_mask].cpu().numpy()
                person_classes = cls[valid_mask]

                for box, conf, cls_id in zip(person_boxes, person_confs, person_classes):
                    detections.append([box[0], box[1], box[2], box[3], conf, cls_id])
                    track_ids.append(-1)  # Placeholder since no tracking
                    class_ids.append(cls_id)

        return np.array(detections, dtype=np.float32) if detections else np.zeros((0, 6), dtype=np.float32), track_ids, class_ids

    except Exception as e:
        print(f"Error in YOLO detection: {e}")
        return np.zeros((0, 6), dtype=np.float32), [], []

def preprocess_bboxes(bboxes, screen_width, screen_height):
    """Vectorized bbox preprocessing"""
    if len(bboxes) == 0:
        return bboxes
        
    bboxes = bboxes.copy()
    bboxes[:, 0] = np.maximum(0, bboxes[:, 0])
    bboxes[:, 1] = np.maximum(0, bboxes[:, 1])
    bboxes[:, 2] = np.minimum(screen_width, bboxes[:, 2])
    bboxes[:, 3] = np.minimum(screen_height, bboxes[:, 3])
    
    valid_mask = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
    
    return bboxes[valid_mask] if valid_mask.any() else np.array([]).reshape(0, bboxes.shape[1])

def extract_camera_id(video_path):
    """Extract camera ID from video filename"""
    filename = os.path.basename(video_path)
    if filename == "Camera.mp4":
        return "Camera"
    elif filename.startswith("Camera_"):
        return filename.replace(".mp4", "")
    else:
        return filename.replace(".mp4", "")

class FrameBuffer:
    """Buffer for processing multiple frames together"""
    def __init__(self, buffer_size=8):
        self.buffer_size = buffer_size
        self.frames = deque(maxlen=buffer_size)
        self.frame_data = deque(maxlen=buffer_size)
    
    def add_frame(self, frame, frame_id, detections, track_ids, class_ids):
        self.frames.append(frame)
        self.frame_data.append({
            'frame_id': frame_id,
            'detections': detections,
            'track_ids': track_ids,
            'class_ids': class_ids
        })
    
    def is_full(self):
        return len(self.frames) == self.buffer_size
    
    def is_empty(self):
        return len(self.frames) == 0
    
    def get_batch(self):
        frames = list(self.frames)
        data = list(self.frame_data)
        self.frames.clear()
        self.frame_data.clear()
        return frames, data

def process_single_video_optimized(video_path, models, args):
    """Optimized video processing with batching and threading"""
    yolo_model, reid_model, orientation_model, pose_model = models
    
    print(f"Processing video: {video_path}")
    
    # Extract camera info
    camera_id = extract_camera_id(video_path)
    warehouse_id = os.path.basename(os.path.dirname(os.path.dirname(video_path)))
    
    # Load video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return None
    
    # Get video info
    screen_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    screen_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {screen_width}x{screen_height}, {total_frames} frames, {fps} FPS")
    
    # Storage for results
    tracklets = []
    all_features = []
    feature_mapping = []
    
    # Frame buffer for batch processing
    frame_buffer = FrameBuffer(buffer_size=16)
    
    # Process frames
    frame_id = 0
    with tqdm(total=total_frames, desc=f"Processing {camera_id}") as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                # Process remaining frames in buffer
                if not frame_buffer.is_empty():
                    process_frame_batch(frame_buffer, models, args, warehouse_id, camera_id, 
                                      screen_width, screen_height, tracklets, all_features, feature_mapping)
                break
            
            # Detect objects with tracking
            detections, track_ids, class_ids = yolo_detect_optimized(yolo_model, frame, conf_thresh=args.conf_thresh)
            
            if len(detections) > 0:
                # Ensure detections is a NumPy array before preprocessing
                detections = np.array(detections, dtype=np.float32)
                # Preprocess bounding boxes
                processed_bboxes = preprocess_bboxes(detections, screen_width, screen_height)
                
                if len(processed_bboxes) > 0:
                    frame_buffer.add_frame(frame, frame_id, processed_bboxes, track_ids, class_ids)
            
            # Process batch when buffer is full
            if frame_buffer.is_full():
                process_frame_batch(frame_buffer, models, args, warehouse_id, camera_id, 
                                  screen_width, screen_height, tracklets, all_features, feature_mapping)
            
            frame_id += 1
            pbar.update(1)
    
    video.release()
    torch.cuda.empty_cache()  # Clear GPU memory
    
    return {
        'tracklets': tracklets,
        'features': np.array(all_features) if all_features else np.array([]),
        'feature_mapping': feature_mapping,
        'video_info': {
            'warehouse_id': warehouse_id,
            'camera_id': camera_id,
            'width': screen_width,
            'height': screen_height,
            'total_frames': total_frames,
            'fps': fps
        }
    }
    
def process_frame_batch(frame_buffer, models, args, warehouse_id, camera_id, 
                       screen_width, screen_height, tracklets, all_features, feature_mapping):
    """Process a batch of frames together for better efficiency"""
    yolo_model, reid_model, orientation_model, pose_model = models
    
    frames, frame_data = frame_buffer.get_batch()
    if not frames:
        return
    
    # Prepare batch data
    all_bboxes = []
    all_track_ids = []
    all_class_ids = []
    
    for data in frame_data:
        frame_detections = []
        for i, (bbox, track_id, cls_id) in enumerate(zip(data['detections'], data['track_ids'], data['class_ids'])):
            frame_detections.append({
                'bbox': bbox[:4],
                'confidence': bbox[4],
                'track_id': track_id,
                'class_id': cls_id
            })
        all_bboxes.append(np.array([det['bbox'] for det in frame_detections], dtype=np.float32) if frame_detections else np.array([]).reshape(0, 4))
        all_track_ids.append([det['track_id'] for det in frame_detections])
        all_class_ids.append([det['class_id'] for det in frame_detections])
    
    # Extract features in batch for all classes
    if frames and any(len(bboxes) > 0 for bboxes in all_bboxes):
        with torch.no_grad():
            reid_features_batch = reid_model.process_frame_batch_optimized(frames[0], all_bboxes)
    else:
        reid_features_batch = [np.array([]).reshape(0, -1) for _ in frames]
    
    # Process each frame in the batch
    for i, (frame, data) in enumerate(zip(frames, frame_data)):
        frame_id = data['frame_id']
        detections = data['detections']
        track_ids = data['track_ids']
        class_ids = data['class_ids']
        
        frame_detections = []
        for j, (bbox, track_id, cls_id) in enumerate(zip(detections, track_ids, class_ids)):
            frame_detections.append({
                'bbox': bbox[:4],
                'confidence': bbox[4],
                'track_id': track_id,
                'class_id': cls_id
            })
        
        if not frame_detections:
            continue
        
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract crops for orientation (batch processing, for all classes)
        crops = []
        crop_indices = []
        for j, detection in enumerate(frame_detections):
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame_rgb[y1:y2, x1:x2]
            crops.append(crop)
            crop_indices.append(j)

        
        # Get orientations in batch
        orientation_results = [None] * len(frame_detections)
        if orientation_model and any(crop is not None for crop in crops):
            valid_crops = [crop for crop in crops if crop is not None]
            valid_crop_indices = [j for j, crop in enumerate(crops) if crop is not None]
            if valid_crops:
                batch_orientations = orientation_model.get_yaw_angles_batch(valid_crops)
                # if len(batch_orientations) != len(valid_crops):
                    # print(f"Warning: Orientation result count ({len(batch_orientations)}) does not match valid crop count ({len(valid_crops)}) for frame {frame_id}")
                for j, orientation in zip(valid_crop_indices, batch_orientations):
                    orientation_results[j] = orientation
        
        # Estimate poses in batch (for all classes)
        bboxes = [det['bbox'] for det in frame_detections]
        pose_results = pose_model.estimate_poses_batch(frame_rgb, [np.array(bboxes, dtype=np.float32)] if bboxes else [np.array([]).reshape(0, 4)])[0] if bboxes else []
        
        # Ensure pose results align with detections
        if len(pose_results) != len(frame_detections):
            # print(f"Warning: Pose result count ({len(pose_results)}) does not match detection count ({len(frame_detections)}) for frame {frame_id}")
            pose_results.extend([{'keypoints': [], 'scores': []}] * (len(frame_detections) - len(pose_results)))
        
        # Process each detection
        reid_features = reid_features_batch[i] if i < len(reid_features_batch) else np.array([]).reshape(0, -1)
        
        # Ensure reid_features align with detections
        if len(reid_features) != len(frame_detections):
            # print(f"Warning: ReID feature count ({len(reid_features)}) does not match detection count ({len(frame_detections)}) for frame {frame_id}")
            reid_features = np.pad(reid_features, ((0, max(0, len(frame_detections) - len(reid_features))), (0, 0)), mode='constant')
        
        # Log detection details for debugging
        # print(f"Frame {frame_id}: {len(frame_detections)} detections, {len(reid_features)} features, {len(pose_results)} poses, {len([r for r in orientation_results if r is not None])} orientations")
        
        for j, detection in enumerate(frame_detections):
            bbox = detection['bbox']
            track_id = detection['track_id']
            class_id = detection['class_id']
            
            # Assign orientation
            orientation_info = orientation_results[j] if j < len(orientation_results) else None
            
            # Assign keypoints
            keypoints_dict = {}
            trajectory_center = None
            if j < len(pose_results):
                pose = pose_results[j]
                keypoints = pose['keypoints']
                scores = pose['scores']
                for k, (kp, score) in enumerate(zip(keypoints, scores)):
                    if k < len(pose_model.keypoint_names):
                        keypoints_dict[pose_model.keypoint_names[k]] = {
                            'x': float(kp[0].cpu().numpy()),
                            'y': float(kp[1].cpu().numpy()),
                            'score': float(score.cpu().numpy())
                        }
                trajectory_center = pose_model.get_trajectory_center(keypoints, scores, bbox)
            
            # Assign ReID feature
            feature_idx = -1
            if j < len(reid_features) and reid_features[j].size > 0:
                feature_idx = len(all_features)
                all_features.append(reid_features[j])
                feature_mapping.append({
                    'warehouse_id': warehouse_id,
                    'camera_id': camera_id,
                    'obj_id': track_id,
                    'frame_id': frame_id,
                    'feature_idx': feature_idx  # Add feature_idx to mapping
                })
            
            # Create tracklet entry
            tracklet = {
                'warehouse_id': warehouse_id,
                'camera_id': camera_id,
                'class_id': int(class_id),
                'obj_id': int(track_id),
                'frame_id': int(frame_id),
                'bbox_2d': {
                    'x1': float(bbox[0]),
                    'y1': float(bbox[1]),
                    'x2': float(bbox[2]),
                    'y2': float(bbox[3]),
                    'confidence': float(detection['confidence'])
                },
                'yaw': {
                    'yaw_angle': float(orientation_info['yaw_angle']) if orientation_info and 'yaw_angle' in orientation_info else None,
                    'yaw_degrees': float(orientation_info['yaw_degrees']) if orientation_info and 'yaw_degrees' in orientation_info else None,
                    'confidence': float(orientation_info['confidence']) if orientation_info and 'confidence' in orientation_info else None
                } if orientation_info else None,
                'keypoints': {
                    key: {
                        'x': float(val['x']),
                        'y': float(val['y']),
                        'score': float(val['score'])
                    } for key, val in keypoints_dict.items()
                },
                'trajectory_center': [float(coord) for coord in trajectory_center] if trajectory_center is not None else None,
                'feature_idx': int(feature_idx)
            }
            tracklets.append(tracklet)

def find_warehouse_videos(base_path):
    """Find all warehouse video files"""
    warehouses = {}
    
    # Look for warehouse directories
    warehouse_pattern = os.path.join(base_path, "Warehouse_*")
    warehouse_dirs = glob.glob(warehouse_pattern)
    
    for warehouse_dir in warehouse_dirs:
        warehouse_id = os.path.basename(warehouse_dir)
        video_dir = os.path.join(warehouse_dir, "videos")
        
        if os.path.exists(video_dir):
            # Find all video files
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov']:
                video_files.extend(glob.glob(os.path.join(video_dir, ext)))
            
            if video_files:
                warehouses[warehouse_id] = sorted(video_files)
    
    return warehouses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, 
                       default='/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/data_aic2025/raw/test',
                       help='Base path containing warehouse directories')
    parser.add_argument('--yolo_model', type=str, 
                       default='/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/aic2025/weights/yolo12x.pt',
                       help='Path to YOLO model')
    parser.add_argument('--reid_model', type=str, 
                       default='/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/aic2025/fast-reid/aic24.pkl',
                       help='Path to ReID model')
    parser.add_argument('--orientation_model', type=str,
                       default='/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/aic2025/OrientAnything/checkpoint9.pth',
                       help='Path to orientation model')
    parser.add_argument('--output_dir', type=str, default='warehouse_tracklets_new',
                       help='Output directory for results')
    parser.add_argument('--conf_thresh', type=float, default=0.1,
                       help='Detection confidence threshold')
    parser.add_argument('--warehouses', type=str, nargs='*',
                       help='Specific warehouses to process (e.g., Warehouse_017 Warehouse_018)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Scanning for warehouse videos...")
    warehouses = find_warehouse_videos(args.base_path)
    
    if not warehouses:
        print("No warehouse videos found!")
        return
    
    if args.warehouses:
        warehouses = {k: v for k, v in warehouses.items() if k in args.warehouses}
    
    print(f"Found {len(warehouses)} warehouses to process:")
    for warehouse_id, videos in warehouses.items():
        print(f"  {warehouse_id}: {len(videos)} videos")
    
    print("Loading models...")
    
    # YOLO model
    if not os.path.exists(args.yolo_model):
        print(f"Error: YOLO model not found: {args.yolo_model}")
        return
    yolo_model = YOLO(args.yolo_model)
    
    # ReID model
    if not os.path.exists(args.reid_model):
        print(f"Error: ReID model not found: {args.reid_model}")
        return
    reid = torch.load(args.reid_model, map_location='cuda:1').to('cuda:1').eval()

    reid_model = OptimizedReidInferencer(reid, batch_size=32)
    orientation_model = FastOrientationEstimator(args.orientation_model) 
    pose_model = FastPoseEstimator() 
    
    models = (yolo_model, reid_model, orientation_model, pose_model)
    
    # Process each warehouse
    for warehouse_id, video_files in warehouses.items():
        print(f"\n{'='*50}")
        print(f"Processing {warehouse_id}")
        print(f"{'='*50}")
        
        warehouse_output_dir = os.path.join(args.output_dir, warehouse_id)
        os.makedirs(warehouse_output_dir, exist_ok=True)
        
        warehouse_tracklets = []
        warehouse_features = []
        warehouse_feature_mapping = []
        feature_offset = 0  
        
        # Inside the warehouse loop
        for video_path in video_files:
            try:
                result = process_single_video_optimized(video_path, models, args) 
                
                if result:
                    for tracklet in result['tracklets']:
                        if tracklet['feature_idx'] != -1:
                            tracklet['feature_idx'] += feature_offset
                    for mapping in result['feature_mapping']:
                        if mapping['feature_idx'] != -1:
                            mapping['feature_idx'] += feature_offset
                    warehouse_tracklets.extend(result['tracklets'])
                    warehouse_feature_mapping.extend(result['feature_mapping'])
                    
                    # Concatenate features
                    if len(result['features']) > 0:
                        if len(warehouse_features) == 0:
                            warehouse_features = result['features']
                        else:
                            warehouse_features = np.concatenate([warehouse_features, result['features']], axis=0)
                        feature_offset += len(result['features'])
                    
                    print(f"  Processed {len(result['tracklets'])} tracklets from {os.path.basename(video_path)}")
                
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
        
        # Save warehouse results
        if warehouse_tracklets:
            # Save tracklets as JSON
            tracklets_path = os.path.join(warehouse_output_dir, 'tracklets.json')
            with open(tracklets_path, 'w') as f:
                json.dump(warehouse_tracklets, f, indent=2, cls=NumpyEncoder)            
            # Save features as NPY
            if len(warehouse_features) > 0:
                features_path = os.path.join(warehouse_output_dir, 'features.npy')
                np.save(features_path, warehouse_features)
                
                # Save feature mapping
                mapping_path = os.path.join(warehouse_output_dir, 'feature_mapping.json')
                with open(mapping_path, 'w') as f:
                    json.dump(warehouse_feature_mapping, f, indent=2, cls=NumpyEncoder)     
            print(f"\n{warehouse_id} Results:")
            print(f"  Total tracklets: {len(warehouse_tracklets)}")
            print(f"  Total features: {len(warehouse_features)}")
            print(f"  Saved to: {warehouse_output_dir}")
        else:
            print(f"No tracklets found for {warehouse_id}")
    
    print(f"\nProcessing complete! Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
    