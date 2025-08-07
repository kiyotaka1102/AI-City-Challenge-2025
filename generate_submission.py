import os
import argparse
from collections import defaultdict
import numpy as np
def load_tracks_for_submission(track_file, scene_id):
    """Load tracking data and convert to submission format, skipping duplicates."""
    tracks = []
    seen_ids = set()  # Track unique (scene_id, frame_id, track_id) combinations
    
    with open(track_file, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            
            parts = line.strip().split(',')
            if len(parts) < 18:
                print(f"Warning: Skipping invalid line with {len(parts)} fields: {line.strip()}")
                continue
            
            try:
                cam_id, track_id, frame_id, class_id, x, y, w, h, gx, gy, gz, w_3d, h_3d, l_3d, yaw_3d, x_center, y_center, z_center = map(float, parts[:18])
                values = [cam_id, track_id, frame_id, class_id, x, y, w, h, gx, gy, gz, w_3d, h_3d, l_3d, yaw_3d, x_center, y_center, z_center]
                if any(not np.isfinite(v) for v in values):  # Skip if any value is NaN or infinite
                    print(f"Skipping line with NaN or infinite values: {line.strip()}")
                    continue
                # Convert to integers where needed
                cam_id = int(cam_id)
                track_id = int(track_id)
                frame_id = int(frame_id)
                class_id = int(class_id)
                
                # Skip invalid entries
                if w_3d <= 0 or h_3d <= 0 or l_3d <= 0:
                    continue
                
                # Check for duplicates based on (scene_id, frame_id, track_id)
                track_key = (scene_id, frame_id, track_id)
                if track_key in seen_ids:
                    print(f"Skipping duplicate track: scene_id={scene_id}, frame_id={frame_id}, track_id={track_id}")
                    continue
                seen_ids.add(track_key)
                
                # Optimize dimensions based on class_id
                if class_id == 1:  # Forklift
                    adjusted_width = min(w_3d + 1.2, 2.6)
                    adjusted_length = min(l_3d + 0.6, 1.8)
                    adjusted_height = min(h_3d, 2.3)
                elif class_id == 2:  # NovaCarter
                    adjusted_width = min(w_3d, 0.75)
                    adjusted_length = min(l_3d, 0.47)
                    adjusted_height = min(h_3d, 0.37)
                    yaw_3d += np.pi/2

                elif class_id == 3:  # Transporter
                    adjusted_width = min(w_3d+0.33, 1.43)
                    adjusted_length = min(l_3d + 0.23, 0.65)
                    adjusted_height = min(h_3d, 0.2)
                    yaw_3d += np.pi/2
                elif class_id ==5:
                    adjusted_width = 0.40
                    adjusted_length = 0.8
                    adjusted_height= 1.8
                else: 
                    adjusted_width = min(w_3d, 0.4)
                    adjusted_length = min(l_3d+0.1,0.8)
                    adjusted_height = min(h_3d, 1.9)
                    
                tracks.append({
                    'cam_id': cam_id,
                    'track_id': track_id,
                    'frame_id': frame_id,
                    'class_id': class_id,
                    # 'x_center': gx if class_id not in [1,2,3,4] else x_center,
                    # 'y_center': gy if class_id  not in [1,2,3,4] else y_center,
                    'x_center':x_center,
                    'y_center':y_center,
                    # 'z_center': adjusted_height/2 if class_id in [0,4,5] else z_center, #previous use this
                    'z_center': adjusted_height/2,
                    'width': adjusted_width if class_id in [3,1,5] else w_3d,
                    'length': adjusted_length if class_id in [3,1,5] else l_3d, 
                    'height': adjusted_height if class_id in [3,1,5] else h_3d,
                    'yaw': yaw_3d
                })
                
            except ValueError as e:
                print(f"Warning: Error parsing line: {line.strip()}. Error: {e}")
                continue
    
    return tracks

def map_class_id(original_class_id):
    """
    Map your class IDs to the required submission format.
    Adjust this mapping based on your actual class definitions.
    
    Required mapping:
    Person→0, Forklift→1, NovaCarter→2, Transporter→3, FourierGR1T2→4, AgilityDigit→5
    """
    # You need to adjust this mapping based on your actual class IDs
    class_mapping = {
        0: 0,  # Person
        1: 1,  # Forklift  
        2: 2,  # NovaCarter
        3: 3,  # Transporter
        4: 4,  # FourierGR1T2
        5: 5,  # AgilityDigit
        # Add more mappings as needed
    }
    
    return class_mapping.get(original_class_id, original_class_id)

def generate_submission_file(track_file, output_file, scene_id=1):
    """
    Generate submission file in the required format.
    
    Format: <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
    """
    tracks = load_tracks_for_submission(track_file, scene_id)
    
    if not tracks:
        print("No valid tracks found!")
        return
    
    # Group tracks by class_id and track_id to ensure unique object_ids per class
    class_track_mapping = defaultdict(dict)
    
    # Create unique object_ids per class
    for track in tracks:
        class_id = map_class_id(track['class_id'])
        track_id = track['track_id']
        
        if track_id not in class_track_mapping[class_id]:
            # Assign a new unique object_id for this class
            class_track_mapping[class_id][track_id] = len(class_track_mapping[class_id]) + 1
    
    # Write submission file
    with open(output_file, 'w') as f:
        for track in tracks:
            mapped_class_id = map_class_id(track['class_id'])
            object_id = class_track_mapping[mapped_class_id][track['track_id']]
            
            # Format: scene_id class_id object_id frame_id x y z width length height yaw
            line = f"{scene_id} {mapped_class_id} {object_id} {track['frame_id']} " \
                   f"{track['x_center']:.2f} {track['y_center']:.2f} {track['z_center']:.2f} " \
                   f"{track['width']:.2f} {track['length']:.2f} {track['height']:.2f} {track['yaw']:.2f}\n"
            f.write(line)
    
    print(f"Submission file generated: {output_file}")
    print(f"Total detections: {len(tracks)}")
    
    # Print statistics
    class_stats = defaultdict(int)
    for track in tracks:
        mapped_class_id = map_class_id(track['class_id'])
        class_stats[mapped_class_id] += 1
    
    print("Class distribution:")
    class_names = {0: "Person", 1: "Forklift", 2: "NovaCarter", 3: "Transporter", 4: "FourierGR1T2", 5: "AgilityDigit"}
    for class_id, count in sorted(class_stats.items()):
        class_name = class_names.get(class_id, f"Class_{class_id}")
        print(f"  {class_name} (ID {class_id}): {count} detections")

def auto_detect_warehouses(base_dir):
    """
    Automatically detect warehouse folders and create scene mapping using warehouse number as scene ID.
    
    Args:
        base_dir: Base directory containing warehouse folders
        
    Returns:
        Dict mapping warehouse folder names to scene IDs (derived from folder name, e.g., Warehouse_017 -> 17)
    """
    scene_mapping = {}
    
    if not os.path.exists(base_dir):
        print(f"Error: Base directory does not exist: {base_dir}")
        return scene_mapping
    
    # Get all directories that match the Warehouse_XXX pattern
    warehouse_folders = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith('Warehouse_'):
            try:
                # Extract warehouse number from folder name (e.g., '017' from 'Warehouse_017')
                warehouse_num = int(item.split('_')[1])
                warehouse_folders.append((warehouse_num, item))
            except (IndexError, ValueError):
                print(f"Warning: Invalid warehouse folder name: {item}")
                continue
    
    # Sort by warehouse number for consistent ordering
    warehouse_folders.sort(key=lambda x: x[0])
    
    # Assign scene_id as the warehouse number
    for warehouse_num, folder_name in warehouse_folders:
        scene_id = warehouse_num  # Use the warehouse number as scene_id
        scene_mapping[folder_name] = scene_id
        print(f"Found: {folder_name} -> Scene ID {scene_id}")
    
    return scene_mapping

def generate_multiple_scenes(base_dir, output_file, scene_mapping=None):
    """
    Generate submission file for multiple scenes.
    
    Args:
        base_dir: Base directory containing scene folders
        output_file: Output submission file path
        scene_mapping: Optional dict mapping scene folder names to scene IDs
                      If None, will auto-detect warehouse folders
    """
    # Auto-detect warehouses if no mapping provided
    if scene_mapping is None:
        print("Auto-detecting warehouse folders...")
        scene_mapping = auto_detect_warehouses(base_dir)
        
        if not scene_mapping:
            print("No warehouse folders found!")
            return
    
    all_lines = []
    scene_stats = {}
    
    for scene_folder, scene_id in sorted(scene_mapping.items(), key=lambda x: x[1]):
        track_file = os.path.join(base_dir, scene_folder, 'global_tracks.txt')
        
        if not os.path.exists(track_file):
            print(f"Warning: Track file not found: {track_file}")
            continue
            
        print(f"Processing scene {scene_id}: {scene_folder}")
        tracks = load_tracks_for_submission(track_file, scene_id)
        
        if not tracks:
            print(f"No valid tracks found in {scene_folder}")
            continue
        
        # Group tracks by class_id and track_id for unique object_ids per class
        class_track_mapping = defaultdict(dict)
        
        for track in tracks:
            class_id = map_class_id(track['class_id'])
            track_id = track['track_id']
            
            if track_id not in class_track_mapping[class_id]:
                class_track_mapping[class_id][track_id] = len(class_track_mapping[class_id]) + 1
        
        # Generate lines for this scene
        scene_lines = []
        for track in tracks:
            mapped_class_id = map_class_id(track['class_id'])
            object_id = class_track_mapping[mapped_class_id][track['track_id']]
            
            line = f"{scene_id} {mapped_class_id} {object_id} {track['frame_id']} " \
                   f"{track['x_center']:.2f} {track['y_center']:.2f} {track['z_center']:.2f} " \
                   f"{track['width']:.2f} {track['length']:.2f} {track['height']:.2f} {track['yaw']:.2f}\n"
            scene_lines.append(line)
        
        all_lines.extend(scene_lines)
        scene_stats[scene_folder] = len(scene_lines)
        print(f"  Added {len(scene_lines)} detections from {scene_folder}")
    
    # Write all lines to output file
    with open(output_file, 'w') as f:
        f.writelines(all_lines)
    
    print(f"\nSubmission file generated: {output_file}")
    print(f"Total detections across all scenes: {len(all_lines)}")
    
    # Print detailed statistics
    print("\nDetailed Statistics:")
    for scene_folder, count in scene_stats.items():
        scene_id = scene_mapping[scene_folder]
        print(f"  Scene {scene_id} ({scene_folder}): {count} detections")

def main():
    parser = argparse.ArgumentParser(description="Generate submission file from tracking results.")
    parser.add_argument('--track_file', type=str,
                        default='/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/aic2025/global_tracks/Warehouse_018/global_tracks.txt',
                        help='Path to global_tracks.txt file')
    parser.add_argument('--output_file', type=str,
                        default='/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/aic2025/result/track1.txt',
                        help='Output submission file path')
    parser.add_argument('--scene_id', type=int, default=1,
                        help='Scene ID for single scene processing')
    parser.add_argument('--multiple_scenes', action='store_true',
                        help='Process multiple scenes from base directory')
    parser.add_argument('--base_dir', type=str,
                        default='/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/aic2025/global_tracks_test',
                        help='Base directory containing multiple scene folders')
    
    args = parser.parse_args()
    
    if args.multiple_scenes:
        # Auto-detect all warehouse folders or use custom mapping
        generate_multiple_scenes(args.base_dir, args.output_file)
    else:
        generate_submission_file(args.track_file, args.output_file, args.scene_id)

if __name__ == '__main__':
    main()