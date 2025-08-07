#%%
import h5py
import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from collections import defaultdict
import math


def to_bev(pixels, H, x_origin, y_origin, scale):
    point = np.array([pixels[0], pixels[1], 1])
    ground_point = np.linalg.inv(H) @ point
    ground_point /= ground_point[2]
    x_map = int((ground_point[0] + x_origin) * scale)
    y_map = int((y_origin - ground_point[1]) * scale) 
    return x_map, y_map

def find_best_angle_group(lines, angle_thresh_deg=20):
    best_group = []
    min_std = 1e6
    for i in range(len(lines)):
        ref_angle = lines[i][0]
        group = [line for line in lines if abs(np.rad2deg(line[0] - ref_angle)) < angle_thresh_deg]
        if len(group) >= 2:
            std = np.std([g[0] for g in group])
            if std < min_std:
                min_std = std
                best_group = group
    return best_group
        

def is_near_border(x, y, w, h, margin=10):
    return (
        x < margin or x >= w - margin or
        y < margin or y >= h - margin
    )

#%%
# === Load JSON files ===
with open('/kaggle/input/depth-file/zip/tracklets.json', 'r') as f:
    track_data = json.load(f)

with open('/kaggle/input/depth-file/zip/calibration.json', 'r') as f:
    calib = json.load(f)

# === Extract camera parameters (nếu cần sau này) ===
for sensor in calib['sensors']:
    if sensor['type'] == 'camera' and sensor['id'] == 'Camera_03':
        intrinsic = np.array(sensor['intrinsicMatrix'])
        extrinsic = np.array(sensor['extrinsicMatrix'])
        scaleFactor = sensor['scaleFactor']
        x_origin = sensor['translationToGlobalCoordinates']['x']
        y_origin = sensor['translationToGlobalCoordinates']['y']
        Homo = sensor['homography']
        break

# === Load depth file ===
depth_file = h5py.File('/kaggle/input/depth-file/zip/depth_maps/Camera_03.h5', 'r')

# === Group frame_id -> objects ===
frame_to_objects = defaultdict(list)
for obj in track_data:
    if obj['camera_id'] == 'Camera_03' and obj['class_id'] == 2:
        frame_to_objects[obj['frame_id']].append(obj)

# === Duyệt frame ===
frame_ids = sorted(frame_to_objects.keys())[:9000]

for idx, frame_id in enumerate(frame_ids):
    depth_key = f"distance_to_image_plane_{frame_id:05d}.png"
    if depth_key not in depth_file:
        print(f"⚠️ Key '{depth_key}' not found.")
        continue

    depth_map = depth_file[depth_key][:]  # (H, W) mm
    depth_vis = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    depth_vis = cv.cvtColor(depth_vis, cv.COLOR_GRAY2BGR)

    for obj in frame_to_objects[frame_id]:
        # --- BBOX gốc ---
        gx1, gy1, gx2, gy2 = int(obj['bbox_2d']['x1']), int(obj['bbox_2d']['y1']), int(obj['bbox_2d']['x2']), int(obj['bbox_2d']['y2'])

        # --- BBOX mở rộng để crop ---
        cx1, cy1, cx2, cy2 = gx1 - 10, gy1 - 10, gx2 + 15, gy2 + 15
        offset_x, offset_y = cx1, cy1  # vị trí crop trong depth gốc

        cropped_depth = depth_map[cy1:cy2, cx1:cx2]
        if cropped_depth.size == 0:
            continue


        # === Phân tích histogram depth trong bbox crop ===
        valid_pixels = cropped_depth[cropped_depth > 0]
        if valid_pixels.size == 0:
            continue
        
        # === Lấy depth lớn nhất (background xa nhất) làm threshold ===
        thresh = np.percentile(valid_pixels, 60)


        max_pos_local = np.unravel_index(np.argmax(cropped_depth, axis=None), cropped_depth.shape)
        thresh_local_y, thresh_local_x = max_pos_local
        
        # === Chuyển sang toạ độ gốc trong depth_map ===
        thresh_point = (thresh_local_x + offset_x, thresh_local_y + offset_y)


        # === Mask từ threshold ===
        mask = (cropped_depth > 0) & (cropped_depth < thresh)
        mask = mask.astype(np.uint8) * 255
        
        kernel = np.ones((3, 3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)


        # contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv.contourArea)
        mask_clean = np.zeros_like(mask)
        cv.drawContours(mask_clean, [largest], -1, 255, thickness=cv.FILLED)


       
        print(f"[{frame_id}] thresh={thresh:.1f}, min={np.min(cropped_depth):.1f}, max={np.max(cropped_depth):.1f}, mask%={(np.sum((cropped_depth>0)&(cropped_depth<thresh))/cropped_depth.size)*100:.1f}%")

        hough_vis = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        edges = cv.Canny(mask_clean, 1, 1)

        lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=25, minLineLength=20, maxLineGap=5)
        lines_info = []


        h, w = mask_clean.shape
        margin = 2 

        if lines is not None:
            for line in lines:
                lx1, ly1, lx2, ly2 = line[0]
                
                if is_near_border(lx1, ly1, w, h, margin) or is_near_border(lx2, ly2, w, h, margin):
                    cv.line(hough_vis, (lx1, ly1), (lx2, ly2), (0, 0, 255), 1)
                    continue  # cả 2 điểm đều chạm biên → bỏ
                

                dx, dy = lx2 - lx1, ly2 - ly1
                angle = np.arctan2(dy, dx)
                length = np.hypot(dx, dy)
                lines_info.append((angle, length, (lx1, ly1, lx2, ly2)))
                cv.line(hough_vis, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)


    
        # Nếu không có line nào thì bỏ qua
        if not lines_info:
            continue
        
        # ✅ Nếu chỉ có 1 line thì vẽ luôn line đó
        if len(lines_info) == 1:
            longest_line = lines_info[0][2]
        else:
            # Bước 1: lọc top 80% đường dài nhất
            lengths = [line[1] for line in lines_info]
            length_thresh = np.percentile(lengths, 80)
            long_lines = [line for line in lines_info if line[1] >= length_thresh]
        
            # Bước 2: tìm group có góc gần nhau nhất trong long_lines
            best_group = find_best_angle_group(long_lines)
        
            # Nếu không có group phù hợp, fallback về toàn bộ
            if not best_group:
                best_group = find_best_angle_group(lines_info)
        
            # Cuối cùng: chọn đoạn dài nhất trong group
            if best_group:
                longest_line = max(best_group, key=lambda x: x[1])[2]
            else:
                longest_line = None

        

        # === Vẽ mũi tên yaw ===
        M = cv.moments(mask)
        if longest_line is not None and M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            lx1, ly1, lx2, ly2 = longest_line
            vx, vy = lx2 - lx1, ly2 - ly1
            norm = np.hypot(vx, vy)
            if norm > 0:
                vx /= norm
                vy /= norm
                length = 60
                end = (int(cx + vx * length), int(cy + vy * length))

                # Vẽ trên ảnh crop
                cv.arrowedLine(hough_vis, (cx, cy), end, (0, 255, 255), 2, tipLength=0.2)

                # Vẽ trên ảnh depth full
                start_full = (int(cx + offset_x), int(cy + offset_y))
                end_full = (int(end[0] + offset_x), int(end[1] + offset_y))
                cv.arrowedLine(depth_vis, start_full, end_full, (0, 0, 255), 2, tipLength=0.2)

                start_bev = to_bev(start_full, Homo, x_origin, y_origin, scaleFactor)
                end_bev = to_bev(end_full, Homo, x_origin, y_origin, scaleFactor)

                dx = end_bev[0] - start_bev[0]
                dy = end_bev[1] - start_bev[1]
                yaw_3d = np.arctan2(dx, dy)

               


        # === Vẽ bbox và điểm threshold ===
        cv.rectangle(depth_vis, (gx1 - 10, gy1 - 10), (gx2 + 15, gy2 + 15), (0, 255, 0), 2)
        cv.circle(depth_vis, thresh_point, 5, (0, 255, 255), -1)

    # === Hiển thị mỗi 5 frame ===
    if idx % 5 == 0:
        print(f"[{frame_id}] Yaw: {math.degrees(yaw_3d):.1f}°")
        plt.figure(figsize=(20, 4))

        plt.subplot(1, 4, 1)
        plt.title(f"Depth View {frame_id}")
        plt.imshow(depth_vis[..., ::-1])
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.title("Mask")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.title("Canny Edges")
        plt.imshow(edges, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.title("HoughLinesP")
        plt.imshow(hough_vis[..., ::-1])
        plt.axis('off')

        plt.tight_layout()
        plt.show()

depth_file.close()