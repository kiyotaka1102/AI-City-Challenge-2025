import cv2
import numpy as np
import os
import glob
import json
from collections import defaultdict
import argparse
from util.func_3d_bbox import *

def to_bev(pixels, H, x_origin, y_origin, scale):
    point = np.array([pixels[0], pixels[1], 1])
    ground_point = np.linalg.inv(H) @ point
    ground_point /= ground_point[2]
    x_map = int((ground_point[0] + x_origin) * scale)
    y_map = int((y_origin - ground_point[1]) * scale)  # dùng 33.5 nếu muốn show đúng trên bev map
    return x_map, y_map

def global_to_bev(gx, gy, x_origin, y_origin, scale):
    """Convert global coordinates directly to BEV map coordinates."""
    x_map = int((gx + x_origin) * scale)
    y_map = int((y_origin - gy) * scale)
    return x_map, y_map

def load_tracks(track_file):
    """Load tracking data from global_tracks.txt."""
    tracks = defaultdict(list)
    with open(track_file, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.strip().split(',')
            if len(parts) < 17:
                print(f"Warning: Skipping invalid line with {len(parts)} fields: {line.strip()}")
                continue
            try:
                cam_id, track_id, frame_id, class_id, x, y, w, h, gx, gy, gz, w_3d, h_3d, l_3d, yaw_3d, x_center, y_center, z_center = map(float, parts[:18])
                if not all(np.isfinite([gx, gy, gz, w_3d, h_3d, l_3d, yaw_3d, x_center, y_center, z_center])):
                    print(f"Warning: Skipping track {track_id} due to non-finite values: {line.strip()}")
                    continue
                if w_3d <= 0 or h_3d <= 0 or l_3d <= 0:
                    print(f"Warning: Skipping track {track_id} due to invalid dimensions: w={w_3d}, h={h_3d}, l={l_3d}")
                    continue
                cam_id = int(cam_id)
                track_id = int(track_id)
                frame_id = int(frame_id)
                bbox = [int(x), int(y), int(x + w), int(y + h)]
                global_coords = np.array([gx, gy, gz]).reshape(3, 1)
                dimension = np.array([w_3d, l_3d, h_3d])
                if class_id in [1,2,3,5]:
                    center_3d = np.array([x_center, y_center, z_center])
                else:
                    center_3d = np.array([gx, gy, z_center])
                center_3d = np.array([x_center, y_center, z_center])
                tracks[(cam_id, frame_id)].append((track_id, bbox, global_coords, yaw_3d, dimension, center_3d, class_id))
            except ValueError as e:
                print(f"Warning: Error parsing line: {line.strip()}. Error: {e}")
                continue
    return tracks

def get_video_files(video_dir):
    """Load video files from the video directory."""
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    video_map = {}
    for video_path in video_files:
        filename = os.path.basename(video_path)
        try:
            if filename.startswith('Camera_'):
                cam_id = int(filename.replace('Camera_', '').replace('.mp4', ''))
            elif filename == 'Camera.mp4':
                cam_id = 0
            else:
                print(f"Skipping invalid video file: {filename}")
                continue
            video_map[cam_id] = video_path
        except ValueError:
            print(f"Skipping invalid video file: {filename}")
            continue
    return video_map

def load_calibration_json(calib_file):
    """Load camera calibration and map parameters from JSON file."""
    try:
        with open(calib_file, 'r') as f:
            calib_data = json.load(f)
        cameras = {}
        map_params = {
            'x_origin': 0.0,
            'y_origin': 0.0,
            'scale': 1.0
        }
        for sensor in calib_data.get('sensors', []):
            if sensor['type'] == 'camera':
                cam_id = sensor['id']
                if cam_id == 'Camera':
                    cam_id = 0
                elif cam_id.startswith('Camera_'):
                    cam_id = int(cam_id.replace('Camera_', ''))
                else:
                    continue
                K = np.array(sensor['intrinsicMatrix'])
                E = np.array(sensor['extrinsicMatrix'])
                P = K @ E  # Recompute projection matrix
                H = np.array(sensor.get('homography', np.eye(3)))
                # Extract map parameters from the first camera (assuming shared)
                map_params['x_origin'] = sensor.get('translationToGlobalCoordinates', {}).get('x', 0.0)
                map_params['y_origin'] = sensor.get('translationToGlobalCoordinates', {}).get('y', 0.0)
                map_params['scale'] = sensor.get('scaleFactor', 1.0)
                cameras[cam_id] = {
                    'K': K,
                    'E': E,
                    'P': P,
                    'H': H
                }
        return cameras, map_params
    except Exception as e:
        print(f"Error loading calibration file {calib_file}: {e}")
        return {}, {'x_origin': 0.0, 'y_origin': 0.0, 'scale': 1.0}


def draw_tracks(frame, tracks, camera_params):
    """Draw 2D and 3D bounding boxes on the frame."""
    color_2d = (0, 0, 255)  
    for track_id, bbox, global_coords, yaw_3d, dimension, center_3d, class_id in tracks:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_2d, 1)
        center_2d = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        cv2.circle(frame, center_2d, 1, (0, 0, 255), 3)

        cv2.putText(frame, f"ID: {track_id}, {class_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_2d, 2)
        if camera_params is not None:
            try:
                w_final, l_final, h_final = dimension
                if class_id ==1:
                    adjusted_dimension = np.array([min(w_final+1,2.6),min(l_final+0.6,1.8),min(h_final,2.3)])
                elif class_id ==2:
                    # yaw_3d+=np.pi/2
                    adjusted_dimension = np.array([min(w_final, 0.75), min(l_final,0.47), min(h_final, 0.37)])
                    # center_3d[2]= adjusted_dimension[2]/2 
                elif class_id ==3:
                    # yaw_3d+=np.pi/2
                    adjusted_dimension = np.array([min(w_final+0.3 , 1.43), min(l_final+0.23,0.65), min(h_final, 0.2)])
                    # center_3d[2]= adjusted_dimension[2]/2 
                elif class_id ==5:
                    adjusted_dimension = np.array([0.40,0.85,1.8])
                    # center_3d[2]= adjusted_dimension[2]/2 
                else:
                    adjusted_dimension = np.array([min(w_final, 0.4), min(l_final+0.1,0.85), min(h_final, 1.84)])
                    # center_3d[2]= adjusted_dimension[2]/2 
                center_3d[2]= adjusted_dimension[2]/2 
                plot_3d_box(
                    img=frame,
                    E=camera_params['E'],
                    K=camera_params['K'],
                    yaw=yaw_3d,
                    dimension=dimension if class_id not in [3,1,5] else adjusted_dimension ,
                    center_location=center_3d
                )
            except Exception as e:
                print(f"Error processing track {track_id}: {e}")
    return frame

def draw_bev_tracks(map_img, tracks, cameras, map_params):
    """Draw BEV tracks on the map image using global coordinates directly."""
    bev_frame = map_img.copy()
    color_bev = (0, 0, 255)  # Red for BEV bounding boxes
    arrow_color = (0, 255, 255)  # Yellow for yaw arrows
    x_origin = map_params['x_origin']
    y_origin = map_params['y_origin']
    scale = map_params['scale']
    
    for track_id, _, global_coords, yaw_3d, dimension, center_3d, _ in tracks:
        try:
            # Use global coordinates directly
            gx, gy, gz = center_3d.flatten()
            
            # Convert global coordinates to BEV map coordinates
            x_map, y_map = global_to_bev(gx, gy, x_origin, y_origin, scale)
            
            # Calculate rectangle dimensions
            w, l, _ = dimension
            w_px = w * scale
            l_px = l * scale
            
            # Define rectangle vertices
            rect = np.array([
                [-w_px / 2, -l_px / 2],
                [w_px / 2, -l_px / 2],
                [w_px / 2, l_px / 2],
                [-w_px / 2, l_px / 2]
            ])
            
            # Rotate rectangle based on yaw
            yaw_rad = -yaw_3d
            rot_matrix = np.array([
                [np.cos(yaw_rad), -np.sin(yaw_rad)],
                [np.sin(yaw_rad), np.cos(yaw_rad)]
            ])
            rect_rotated = (rot_matrix @ rect.T).T
            rect_rotated = rect_rotated + np.array([x_map, y_map])
            rect_rotated = rect_rotated.astype(int)
            
            # Draw rectangle
            cv2.polylines(bev_frame, [rect_rotated], isClosed=True, color=color_bev, thickness=2)
            
            # Draw track ID
            cv2.putText(bev_frame, f"ID: {track_id}", (x_map + 10, y_map - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bev, 2)
            
            # Draw yaw arrow
            arrow_length = max(w_px, l_px) * 1.2
            arrow_end = np.array([arrow_length * np.cos(yaw_rad), arrow_length * np.sin(yaw_rad)])
            arrow_end = (arrow_end + np.array([x_map, y_map])).astype(int)
            cv2.arrowedLine(bev_frame, (x_map, y_map), tuple(arrow_end), arrow_color, 2, tipLength=0.3)
            
        except Exception as e:
            print(f"Error drawing BEV track {track_id}: {e}")
    
    return bev_frame


def visualize_tracks(track_file, video_dir, output_dir, calib_file, map_file, start_frame=0000):
    """Visualize 2D/3D bounding boxes and BEV on video frames and save the output."""
    os.makedirs(output_dir, exist_ok=True)
    tracks = load_tracks(track_file)
    video_map = get_video_files(video_dir)
    cameras, map_params = load_calibration_json(calib_file)
    
    if not cameras:
        print("Error: No camera calibrations loaded")
        return
    
    map_img = cv2.imread(map_file)
    if map_img is None:
        print(f"Error: Could not load map image {map_file}")
        return
    
    max_frame_id = max(frame_id for (_, frame_id) in tracks.keys())
    
    bev_output_path = os.path.join(output_dir, "BEV_yaw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    bev_out = cv2.VideoWriter(bev_output_path, fourcc, 30, (map_img.shape[1], map_img.shape[0]))
    
    for cam_id, video_path in sorted(video_map.items()):
        print(f"Processing video for Camera {cam_id}: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            continue
        
        camera_params = cameras.get(cam_id)
        if camera_params is None:
            print(f"Warning: No calibration found for Camera {cam_id}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        output_path = os.path.join(output_dir, f"Camera_{cam_id:02d}_yaw2d.mp4" if cam_id != 0 else "Camera_yaw2d.mp4")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_id = start_frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id >300:
                break
            
            frame_tracks = tracks.get((cam_id, frame_id), [])
            if frame_tracks:
                frame = draw_tracks(frame, frame_tracks, camera_params)
            
            out.write(frame)
            
            # Generate BEV frame for all cameras at this frame
            bev_frame_tracks = []
            for c_id in video_map.keys():
                bev_frame_tracks.extend(tracks.get((c_id, frame_id), []))
            
            bev_frame = draw_bev_tracks(map_img, bev_frame_tracks, cameras, map_params)
            bev_out.write(bev_frame)
            
            frame_id += 1
            if frame_id % 100 == 0:
                print(f"Camera {cam_id}: Processed {frame_id}/{total_frames} frames")
        
        cap.release()
        out.release()
        print(f"Finished processing Camera {cam_id}. Output saved to {output_path}")
    
    # Handle remaining frames for BEV (in case some cameras have fewer frames)
    for frame_id in range(frame_id, max_frame_id + 1):
        bev_frame_tracks = []
        for c_id in video_map.keys():
            bev_frame_tracks.extend(tracks.get((c_id, frame_id), []))
        
        bev_frame = draw_bev_tracks(map_img, bev_frame_tracks, cameras, map_params)
        bev_out.write(bev_frame)
        
        if frame_id % 100 == 0:
            print(f"BEV: Processed {frame_id}/{max_frame_id} frames")
    
    bev_out.release()
    print(f"Finished processing BEV. Output saved to {bev_output_path}")
    print(f"Visualization complete! Output videos saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Visualize 2D/3D bounding boxes and BEV.")
    parser.add_argument('--track_file', type=str,
                        default='/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/aic2025/global_tracks_test/Warehouse_019/global_tracks.txt',
                        help='Path to global_tracks.txt file')
    parser.add_argument('--video_dir', type=str,
                        default='/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/data_aic2025/raw/test/Warehouse_019/videos',
                        help='Directory containing video files')
    parser.add_argument('--output_dir', type=str,
                        default='/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/aic2025/visualized_tracks/Warehouse_019',
                        help='Output directory for visualized videos')
    parser.add_argument('--calib_file', type=str,
                        default='/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/data_aic2025/raw/test/Warehouse_019/calibration.json',
                        help='Path to camera calibration JSON file')
    parser.add_argument('--map_file', type=str,
                        default='/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/data_aic2025/raw/test/Warehouse_019/map.png',
                        help='Path to map image for BEV visualization')
    
    args = parser.parse_args()
    visualize_tracks(args.track_file, args.video_dir, args.output_dir, args.calib_file, args.map_file)

if __name__ == '__main__':
    main()
