#%%
import h5py
import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def to_bev(pixels, H, x_origin, y_origin, scale):
    point = np.array([pixels[0], pixels[1], 1])
    ground_point = np.linalg.inv(H) @ point
    ground_point /= ground_point[2]
    x_map = int((ground_point[0] + x_origin) * scale)
    y_map = int((y_origin - ground_point[1]) * scale) 
    return x_map, y_map



def project_3dcamera_to_2d(point_3d, intrinsics):
    X, Y, Z = point_3d
    if abs(Z) < 1e-6:  # Tránh chia cho 0
        return None
    u = (intrinsics['fx'] * X / Z) + intrinsics['cx']
    v = (intrinsics['fy'] * Y / Z) + intrinsics['cy']
    return int(u), int(v)

def calculate_3d_yaw(mask, cropped_depth, camera_intrinsics, x1, y1):
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics["fy"]
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']

    v_mask, u_mask = np.where(mask > 0)
    if len(v_mask) < 20:
        print("Return vì quá ít mask>")
        return (None,)*7

    
    min_v, max_v = np.min(v_mask), np.max(v_mask)
    height = max_v - min_v
    if height == 0:
        return None, None, None, None, None, None, None
    
    upper_body_v_start = min_v
    upper_body_v_end = min_v + int(height * 0.5)

    upper_body_mask = np.zeros_like(mask)
    upper_body_mask[upper_body_v_start  : upper_body_v_end, :] = mask[upper_body_v_start : upper_body_v_end, :]

    v, u = np.where(upper_body_mask > 0)
    d = cropped_depth[v, u]
    valid_indices = d > 0
    u, v, d = u[valid_indices], v[valid_indices], d[valid_indices]

    if len(u) < 20:
        print("Return vì quá ít depth>0>")
        return (None,)*7
    

    
    crop_h, crop_w = cropped_depth.shape
    cx_crop = cx - x1  # Điều chỉnh dựa trên tọa độ ảnh gốc
    cy_crop = cy - y1

    x = (u - cx_crop) * d /fx
    y = (v - cy_crop) * d /fy
    z = d


    points_3d = np.vstack((x, y, z)).T

    # Lọc nhiễu dựa trên độ sâu
    mean_z = np.mean(z)
    std_z = np.std(z)
    z_threshold = 2.0  # Số lần độ lệch chuẩn
    z_mask = (z >= mean_z - z_threshold * std_z) & (z <= mean_z + z_threshold * std_z)
    points_3d = points_3d[z_mask]

    if points_3d.shape[0] < 50:
        return None, None, None

    # Lọc nhiễu dựa trên khoảng cách
    centroid = np.mean(points_3d, axis=0)
    distances = np.linalg.norm(points_3d - centroid, axis=1)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    dist_threshold = 2.0  # Số lần độ lệch chuẩn
    dist_mask = distances <= mean_dist + dist_threshold * std_dist
    points_3d = points_3d[dist_mask]

    if points_3d.shape[0] < 20:
        print("Return vì quá ít point cloud>")
        return (None,)*7
    
     # Chạy PCA

    mean, eigenvectors, _ = cv.PCACompute2(np.float32(points_3d), mean=np.empty((0)))

    normal_vector = eigenvectors[2]
    centroid_3d = mean[0]

      # Điều chỉnh hướng vector pháp tuyến
    if np.dot(normal_vector, centroid_3d) > 0:
        normal_vector = -normal_vector

    # Tính điểm đầu và điểm cuối của vector yaw trong 3D
    k = 250.0  # Độ dài vector (mm), khớp với length=250 trong biểu đồ 3D
    P1 = centroid_3d  # Điểm đầu
    P2 = centroid_3d + k * normal_vector  # Điểm cuối (dùng cả 3 thành phần Nx, Ny, Nz)


    # Chiếu sang 2D (trên ảnh cắt)
    intrinsics_crop = {'fx': fx, 'fy': fy, 'cx': cx_crop, 'cy': cy_crop}
    start_point_2d_crop = project_3dcamera_to_2d(P1, intrinsics_crop)
    end_point_2d_crop = project_3dcamera_to_2d(P2, intrinsics_crop)

    # Chiếu sang 2D (trên ảnh gốc)
    intrinsics_full = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
    start_point_2d_full = project_3dcamera_to_2d(P1, intrinsics_full)
    end_point_2d_full = project_3dcamera_to_2d(P2, intrinsics_full)

    if start_point_2d_crop is None or end_point_2d_crop is None or \
       start_point_2d_full is None or end_point_2d_full is None:
        print('loi day ne')
        return (None,)*7

    return start_point_2d_crop, end_point_2d_crop, start_point_2d_full, end_point_2d_full, points_3d, normal_vector, centroid_3d

#%%
# === LOAD DATA ===
with open('/kaggle/input/depth-file/zip/tracklets.json', 'r') as f:
    track_data = json.load(f)

with open('/kaggle/input/depth-file/zip/calibration.json', 'r') as f:
    calib = json.load(f)

for sensor in calib['sensors']:
    if sensor['type'] == 'camera' and sensor['id'] == 'Camera_02':
        extrinsic = np.array(sensor['extrinsicMatrix'])
        intrinsic = np.array(sensor['intrinsicMatrix'])
        scaleFactor = sensor['scaleFactor']
        x_origin = sensor['translationToGlobalCoordinates']['x']
        y_origin = sensor['translationToGlobalCoordinates']['y']
        Homo = sensor['homography']
        camera_intrinsics = {
            'fx': intrinsic[0, 0], 'fy': intrinsic[1, 1],
            'cx': intrinsic[0, 2], 'cy': intrinsic[1, 2]
        }
        break

depth_file = h5py.File('/kaggle/input/depth-file/zip/depth_maps/Camera_02.h5', 'r')

frame_to_objects = defaultdict(list)
for obj in track_data:
    if obj['camera_id'] == 'Camera_02' and obj['class_id'] == 5:
        frame_to_objects[obj['frame_id']].append(obj)

frame_ids = sorted(frame_to_objects.keys())[:9000]


output_video_path = "yaw_white_robot_cam2_no_filter.avi"
frame_width, frame_height = 1920, 540  # điều chỉnh theo nhu cầu
fps = 10  # tốc độ khung hình

fourcc = cv.VideoWriter_fourcc(*'MJPG')  # Codec cho .avi
video_writer = cv.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


for idx, frame_id in enumerate(frame_ids):
    depth_key = f"distance_to_image_plane_{frame_id:05d}.png"
    if depth_key not in depth_file:
        continue

    depth_map = depth_file[depth_key][:]
    depth_vis_full = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    depth_vis_full = cv.cvtColor(depth_vis_full, cv.COLOR_GRAY2BGR)

    for obj in frame_to_objects[frame_id]:
        gx1, gy1 , gx2, gy2 = int(obj['bbox_2d']['x1']), int(obj['bbox_2d']['y1']), int(obj['bbox_2d']['x2']), int(obj['bbox_2d']['y2'])
        x1, y1, x2, y2 = gx1, gy1, gx2, gy2
        center_bbox = (int((x1 + x2) / 2), int((y1 + y2) / 2)) 

        cropped_depth = depth_map[y1:y2, x1:x2]
        if cropped_depth.size == 0:
            continue

        H, W = cropped_depth.shape
        patch_height = H // 10
        mask = np.zeros_like(cropped_depth, dtype=np.uint8)
        for i in range(10):
            y_start = i * patch_height
            y_end = H if i == 9 else (i + 1) * patch_height
            patch = cropped_depth[y_start:y_end, :]
            if patch.size == 0: continue
            bottom_row = patch[-1, :]
            if bottom_row.size > 0:
                patch_thresh = np.max(bottom_row)
                patch_mask = (patch > 0) & (patch < patch_thresh)
                mask[y_start:y_end, :] = patch_mask.astype(np.uint8)
        mask *= 255



        
        result = calculate_3d_yaw(mask, cropped_depth, camera_intrinsics, x1, y1)
        if result is None:
            print(f"Frame {frame_id}: Không tính được yaw") # Bỏ comment nếu muốn debug
            continue # Bỏ qua object này và sang object tiếp theo

        (start_point_2d_crop, end_point_2d_crop,
         start_point_2d_full, end_point_2d_full,
         points_3d, normal_vector, centroid_3d) = result


        # compute yaw2d
        
        start_bev = to_bev(start_point_2d_full, Homo, x_origin, y_origin, scaleFactor)
        end_bev = to_bev(end_point_2d_full, Homo, x_origin, y_origin, scaleFactor)

        dx = end_bev[0] - start_bev[0]
        dy = end_bev[1] - start_bev[1]
        yaw_3d = np.arctan2(dx, dy)
       
        # yaw_3d =  compute_yaw3d_from_yaw2d(center_bbox, yaw_2d, depth_map, extrinsic, intrinsic)
        
    
        # Vẽ trên ảnh cắt
        vis_crop = cv.normalize(cropped_depth, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        vis_crop = cv.cvtColor(vis_crop, cv.COLOR_GRAY2BGR)
        mask_vis = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

        v_mask, _ = np.where(mask > 0)
        if len(v_mask) >= 20:
            min_v, max_v = np.min(v_mask), np.max(v_mask)
            height = max_v - min_v
            upper_start = min_v 
            upper_end = min_v + int(height * 0.50)
            upper_mask = np.zeros_like(mask)
            upper_mask[upper_start:upper_end, :] = mask[upper_start:upper_end, :]
            upper_vis = cv.cvtColor(upper_mask, cv.COLOR_GRAY2BGR)
        else:
            upper_vis = np.zeros_like(mask_vis)

        # Vẽ mũi tên trên ảnh cắt
        cv.arrowedLine(upper_vis, start_point_2d_crop, end_point_2d_crop, (0, 0, 255), 2, tipLength=0.3)

        # Vẽ trên ảnh gốc
        cv.arrowedLine(depth_vis_full, start_point_2d_full, end_point_2d_full, (0, 0, 255), 2, tipLength=0.3)
        cv.arrowedLine(depth_vis_full, start_point_2d_full, end_point_2d_full, (0, 0, 255), 2, tipLength=0.3)
        cv.rectangle(depth_vis_full, (x1, y1), (x2, y2), (0, 255, 0), 2)



        # Hiển thị
        if frame_id % 50 == 0:
            print(f"🧭 Frame {frame_id}")

            
            fig = plt.figure(figsize=(18, 5))
            
            # Subplot 1: Depth + BBox + Yaw
            ax1 = fig.add_subplot(1, 4, 1)
            ax1.imshow(depth_vis_full[..., ::-1])
            ax1.set_title(f"Frame {frame_id} / Yaw: {np.rad2deg(yaw_3d)}°")
            ax1.axis('off')
        
            # Subplot 2: Full Mask
            ax2 = fig.add_subplot(1, 4, 2)
            ax2.imshow(mask_vis[..., ::-1])
            ax2.set_title("Full Mask")
            ax2.axis('off')
        
            # Subplot 3: Upper body + Yaw
            ax3 = fig.add_subplot(1, 4, 3)
            ax3.imshow(upper_vis[..., ::-1])
            ax3.set_title("Upper Body + Yaw")
            ax3.axis('off')
        
            # Subplot 4: 3D Point Cloud
            ax4 = fig.add_subplot(1, 4, 4, projection='3d')
            ax4.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=1, c='blue')
            ax4.quiver(centroid_3d[0], centroid_3d[1], centroid_3d[2],
                       normal_vector[0], normal_vector[1], normal_vector[2],
                       length=250, color='red', label='Yaw direction')
            ax4.set_xlabel('X')
            ax4.set_ylabel('Y')
            ax4.set_zlabel('Z')
            ax4.set_title("3D Point Cloud")
        
            plt.tight_layout()
            plt.show()

            
            canvas = FigureCanvas(fig)
            canvas.draw()
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            w, h = fig.get_size_inches() * fig.dpi
            image = buf.reshape(int(h), int(w), 3)
            
            # Resize (nếu cần) để khớp với kích thước video
            image = cv.resize(image, (frame_width, frame_height))
            video_writer.write(cv.cvtColor(image, cv.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV
            plt.close(fig)

video_writer.release()            
depth_file.close()
print("Video đã được ghi vào 'output_yaw_visualization.avi'")