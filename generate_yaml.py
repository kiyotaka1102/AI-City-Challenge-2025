import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import psutil  # For memory monitoring
import logging  # For logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_image_size(frame_path):
    """
    Get image dimensions without loading the full image
    """
    try:
        # Use cv2 to read only metadata
        cap = cv2.VideoCapture(frame_path)
        if not cap.isOpened():
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return height, width
    except Exception as e:
        logging.warning(f"Could not get size for {frame_path}: {e}")
        return None

def extract_frames(video_path, output_dir, warehouse, frame_interval=10):
    """
    Extract frames from video files at specified intervals
    """
    os.makedirs(output_dir, exist_ok=True)
    
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        logging.error(f"Error opening video file: {video_path}")
        return []
    
    frame_paths = []
    camera_name = os.path.basename(video_path).split('.')[0]
    frame_count = 0
    saved_count = 0
    
    while True:
        success, frame = video.read()
        if not success:
            break
            
        if frame_count % frame_interval == 0:
            frame_filename = f"{warehouse}_{camera_name}_frame_{frame_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_paths.append((warehouse, camera_name, frame_count, frame_path))
            saved_count += 1
            
        frame_count += 1
    
    video.release()
    logging.info(f"Extracted {saved_count} frames from {video_path}")
    return frame_paths

def parse_ground_truth(json_path):
    """
    Parse the ground truth JSON file
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Loaded annotations for {len(data)} frames from {json_path}")
        return data
    except Exception as e:
        logging.error(f"Error parsing JSON file {json_path}: {e}")
        return {}

def convert_to_yolo_format(ground_truth, frame_data, class_mapping, output_dir):
    """
    Convert annotations to YOLO format and write directly to files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    frame_indices = {(warehouse, cam, idx): path for warehouse, cam, idx, path in frame_data}
    
    for frame_idx_str, annotations in tqdm(ground_truth.items(), desc="Processing annotations"):
        frame_idx = int(frame_idx_str)
        
        for annotation in annotations:
            if "2d bounding box visible" not in annotation:
                continue
                
            for camera_name, bbox in annotation["2d bounding box visible"].items():
                for warehouse in set(item[0] for item in frame_data):
                    frame_key = (warehouse, camera_name, frame_idx)
                    if frame_key not in frame_indices:
                        continue
                        
                    frame_path = frame_indices[frame_key]
                    img_size = get_image_size(frame_path)
                    if img_size is None:
                        logging.warning(f"Skipping {frame_path} due to invalid size")
                        continue
                        
                    img_height, img_width = img_size
                    
                    object_type = annotation["object type"]
                    if object_type not in class_mapping:
                        continue
                        
                    class_id = class_mapping[object_type]
                    
                    x_min, y_min, x_max, y_max = bbox
                    x_min = max(0, min(x_min, img_width - 1))
                    y_min = max(0, min(y_min, img_height - 1))
                    x_max = max(0, min(x_max, img_width - 1))
                    y_max = max(0, min(y_max, img_height - 1))
                    
                    if x_max <= x_min or y_max <= y_min:
                        continue
                        
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height
                    x_center = (x_min + x_max) / (2 * img_width)
                    y_center = (y_min + y_max) / (2 * img_height)
                    
                    # Write annotation directly to file
                    label_path = frame_path.replace('.jpg', '.txt')
                    with open(label_path, 'a') as label_file:  # Append mode
                        label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        
        # Log memory usage periodically
        process = psutil.Process()
        mem_info = process.memory_info()
        logging.info(f"Memory usage: {mem_info.rss / 1024**2:.2f} MB")

def create_train_val_splits(frame_data, output_dir):
    """
    Create train and validation splits
    """
    train_list_file = open(os.path.join(output_dir, "train.txt"), "w")
    val_list_file = open(os.path.join(output_dir, "val.txt"), "w")
    
    frame_paths = [item[3] for item in frame_data]
    np.random.shuffle(frame_paths)
    split_idx = int(len(frame_paths) * 0.8)
    train_frames = frame_paths[:split_idx]
    val_frames = frame_paths[split_idx:]
    
    for frame_path in train_frames:
        abs_frame_path = os.path.abspath(frame_path)
        train_list_file.write(f"{abs_frame_path}\n")
    
    for frame_path in val_frames:
        abs_frame_path = os.path.abspath(frame_path)
        val_list_file.write(f"{abs_frame_path}\n")
    
    train_list_file.close()
    val_list_file.close()
    logging.info(f"Created train.txt with {len(train_frames)} frames and val.txt with {len(val_frames)} frames")

def create_yaml_config(class_mapping, output_dir):
    """
    Create YAML configuration file for YOLO training
    """
    classes = sorted(class_mapping.items(), key=lambda x: x[1])
    class_names = [c[0] for c in classes]
    
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"path: {output_dir}\n")
        f.write(f"train: train.txt\n")
        f.write(f"val: val.txt\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")
    
    return yaml_path

def process_warehouse(warehouse, data_dir, output_dir, frame_interval, all_frame_data, ground_truth_all):
    """
    Process a single warehouse dataset
    """
    warehouse_path = os.path.join(data_dir, warehouse)
    ground_truth_path = os.path.join(warehouse_path, "ground_truth.json")
    videos_dir = os.path.join(warehouse_path, "videos")
    
    # Process ground truth data
    ground_truth = parse_ground_truth(ground_truth_path)
    if not ground_truth:
        logging.warning(f"Skipping {warehouse} due to invalid ground truth data")
        return
    
    ground_truth_all[warehouse] = ground_truth
    
    # Process video files and extract frames
    if not os.path.exists(videos_dir):
        logging.warning(f"Videos directory {videos_dir} does not exist for {warehouse}")
        return
    
    video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
    
    for video_file in tqdm(video_files, desc=f"Processing videos for {warehouse}"):
        video_path = os.path.join(videos_dir, video_file)
        frame_data = extract_frames(video_path, output_dir, warehouse, frame_interval)
        all_frame_data.extend(frame_data)

def main():
    parser = argparse.ArgumentParser(description="Generate YOLO labels for warehouse datasets")
    parser.add_argument("--data_dir", type=str, default="Warehouse_",
                        help="Directory containing warehouse data")
    parser.add_argument("--output_dir", type=str, default="Warehouse_000",
                        help="Output directory for YOLO dataset")
    parser.add_argument("--frame_interval", type=int, default=30,
                        help="Extract 1 frame per this many frames")
    
    args = parser.parse_args()
    
    # Create output directories
    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Initialize data structures
    all_frame_data = []
    ground_truth_all = {}
    class_mapping = {
        "Person": 0,
        "Forklift": 1,
        "NovaCarter": 2,
        "Transporter": 3,
        "FourierGR1T2": 4,
        "AgilityDigit": 5
    }
    
    # Write class names to a file
    with open(os.path.join(args.output_dir, "classes.txt"), "w") as f:
        for cls_name, cls_id in sorted(class_mapping.items(), key=lambda x: x[1]):
            f.write(f"{cls_id}: {cls_name}\n")
    
    # # Process warehouses from Warehouse_000 to Warehouse_014
    # for i in range(15):  # 0 to 14
    #     warehouse = f"Warehouse_{i:03d}"
    #     logging.info(f"Processing {warehouse}...")
    #     process_warehouse(warehouse, args.data_dir, frames_dir, args.frame_interval, all_frame_data, ground_truth_all)
    
    # Convert annotations to YOLO format for all warehouses
    for warehouse, ground_truth in ground_truth_all.items():
        logging.info(f"Converting annotations for {warehouse}...")
        convert_to_yolo_format(ground_truth, all_frame_data, class_mapping, args.output_dir)
    
    # Create train/val splits
    logging.info("Creating train/val splits...")
    create_train_val_splits(all_frame_data, args.output_dir)
    
    # Create YAML configuration
    yaml_path = create_yaml_config(class_mapping, args.output_dir)
    logging.info(f"Created dataset configuration at {yaml_path}")

if __name__ == "__main__":
    main()