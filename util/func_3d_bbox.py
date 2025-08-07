import numpy as np
import cv2 as cv
from enum import Enum
import math
from scipy.spatial.transform import Rotation as R
import h5py
from scipy.spatial import ConvexHull
import warnings
from scipy.spatial.distance import euclidean
from functools import lru_cache
from sklearn.decomposition import PCA
# ------------------thresholds and constants for person------------------
thresh_shoulder_min = 0.1  
thresh_ankle_move   = 0.3
l_const             = 0.5
w_const             = 0.8

def discrete_frechet(P, Q):
        n, m = len(P), len(Q)
        ca = np.full((n, m), -1.0)

        @lru_cache(None)
        def c(i, j):
            if ca[i, j] > -1:
                return ca[i, j]
            elif i == 0 and j == 0:
                ca[i, j] = euclidean(P[0], Q[0])
            elif i > 0 and j == 0:
                ca[i, j] = max(c(i-1, 0), euclidean(P[i], Q[0]))
            elif i == 0 and j > 0:
                ca[i, j] = max(c(0, j-1), euclidean(P[0], Q[j]))
            elif i > 0 and j > 0:
                ca[i, j] = max(
                    min(c(i-1, j), c(i-1, j-1), c(i, j-1)), 
                    euclidean(P[i], Q[j])
                )
            else:
                ca[i, j] = float("inf")
            return ca[i, j]

        return c(n-1, m-1)
def rotation_matrix(yaw):

    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return R

# option to rotate and shift (for label info)
def create_corners(dimension, location=None, R=None):

    dx = dimension[0] / 2
    dy = dimension[1] / 2
    dz = dimension[2] / 2

    x_corners = [dx, -dx, -dx, dx, dx, -dx, -dx, dx] 
    y_corners = [-dy, -dy, dy, dy, -dy, -dy, dy, dy]
    z_corners = [dz, dz, dz, dz, -dz, -dz, -dz, -dz]

    corners = [x_corners, y_corners, z_corners]

    # rotate if R is passed in
    if R is not None:
        corners = np.dot(R, corners)

    # shift if location is passed in
    if location is not None:
        for i,loc in enumerate(location):
            corners[i,:] = corners[i,:] + loc

    final_corners = []
    for i in range(8):
        final_corners.append([corners[0][i], corners[1][i], corners[2][i]])


    return final_corners


# takes in a 3d point and projects it into 2d
def project_3d_word_to_pixel(points, E, K):
    points = np.array(points)
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    P = K @ E

    pixels = (P @ points_homo.T).T
    pixels_2d = pixels[:, :2] / pixels[:, 2:3]

    return pixels_2d

def project_pixel_to_3dworld(pixel, d, E, K):
    u = pixel[0]
    v = pixel[1]
    pixel = np.array([u, v, 1])
    point_cam = d * (np.linalg.inv(K) @ pixel)
    R = E[:, :3]
    t = E[:, -1]
    point_world = np.linalg.inv(R) @ (point_cam - t)

    return point_world

def to_bev(pixels, H, x_origin, y_origin, scale):
    point = np.array([pixels[0], pixels[1], 1])
    ground_point = np.linalg.inv(H) @ point
    ground_point /= ground_point[2]
    x_map = int((ground_point[0] + x_origin) * scale)
    y_map = int((y_origin - ground_point[1]) * scale) # dùng 33.5 nếu muốn show đúng trên bev map, còn dùng y_origin trong calib thì sẽ bị lệch
    return x_map, y_map



class cv_colors(Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)

def plot_3d_box(img, E, K, yaw, dimension, center_location, yaw_gt=None, object_id=None):

    R_y = rotation_matrix(yaw)
    corners = create_corners(dimension, location = center_location, R = R_y)
    corners = np.array(corners)
    
    pixels_2d = project_3d_word_to_pixel(corners, E, K)
    pixels_2d = pixels_2d.astype(int)
    
    edges = [
        (0,4), (1,5), (2,6), (3,7),
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4)
    ]

    for i, j in edges:
        
        cv.line(img, tuple(pixels_2d[i]), tuple(pixels_2d[j]), cv_colors.GREEN.value, 2)


    for idx, pt in enumerate(pixels_2d):
        cv.putText(img, str(idx), tuple(pt), cv.FONT_HERSHEY_SIMPLEX, 0.5, cv_colors.RED.value, 1)
    center_pixel = project_3d_word_to_pixel(center_location.reshape(1, -1), E, K)
    cv.circle(img, tuple(center_pixel[0].astype(int)), 1,(0, 255, 255), 3)

    bottom_corners = [4,5,6,7]
    bottom_pts = corners[bottom_corners, :]

    bottom_center_3d = np.mean(bottom_pts, axis = 0, keepdims=True).T
    forward_dir = R_y @ np.array([[0], [-1], [0]])
    arrow_tip_3d = bottom_center_3d + 1.0 * forward_dir
    pts_3d = np.hstack((bottom_center_3d, arrow_tip_3d))

    pts_cam = (E @ np.vstack((pts_3d, np.ones((1, 2)))))  # (3,2)
    pts_img = (K @ pts_cam[:3] / pts_cam[2])[:2].T  # (2,2)

    
    pt1 = tuple(pts_img[0].astype(int))
    pt2 = tuple(pts_img[1].astype(int))


    if np.all(np.isfinite(pt1)) and np.all(np.isfinite(pt2)):
        cv.arrowedLine(img, pt1, pt2, (0, 255, 0), 2, tipLength=0.3)
        cv.putText(img, f'{math.degrees(yaw):.1f}deg', pt2, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Compute top center pixel
    top_corners = [0, 1, 2, 3]  # Top face corners (z=top)
    top_center_pixel = np.mean(pixels_2d[top_corners], axis=0).astype(int)

    # Draw object ID
    if object_id is not None:
        cv.putText(
            img,
            f'ID: {object_id}',
            tuple(top_center_pixel),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),  # Yellow
            1
        )

    # --- Draw arrow for yaw_gt (if provided) ---
    if yaw_gt is not None:
        R_gt = rotation_matrix(yaw_gt)
        forward_dir_gt = R_gt @ np.array([[0], [-1], [0]])
        arrow_tip_gt_3d = bottom_center_3d + 0.5 * forward_dir_gt
        pts_gt_3d = np.hstack((bottom_center_3d, arrow_tip_gt_3d))

        pts_gt_cam = (E @ np.vstack((pts_gt_3d, np.ones((1, 2)))))
        pts_gt_img = (K @ pts_gt_cam[:3] / pts_gt_cam[2])[:2].T

        pt1_gt = tuple(pts_gt_img[0].astype(int))
        pt2_gt = tuple(pts_gt_img[1].astype(int))

        if np.all(np.isfinite(pt1_gt)) and np.all(np.isfinite(pt2_gt)):
            cv.arrowedLine(img, pt1_gt, pt2_gt, (0, 0, 255), 2, tipLength=0.3)
            cv.putText(img, f'{math.degrees(yaw_gt):.1f}deg', pt2_gt, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)



    cv.line(img, tuple(pixels_2d[0]), tuple(pixels_2d[5]), cv_colors.BLUE.value, 1)
    cv.line(img, tuple(pixels_2d[1]), tuple(pixels_2d[4]), cv_colors.BLUE.value, 1)


def compute_2d_yaw(camera, location, yaw, object_type):
    # print(f"Computing yaw for location {location.flatten()}, yaw {yaw:.6f} radians, type: {object_type}")

    # Select forward vector based on object type

    forward_local = np.array([0, -1, 0]).reshape(3, 1)

    R_yaw = rotation_matrix(yaw)
    forward_world = R_yaw @ forward_local
    # print(f"Forward vector (world): {forward_world.flatten()}")
    
    tip = location + forward_world * 1.0
    location_hom = np.vstack([location, [[1]]])
    tip_hom = np.vstack([tip, [[1]]])
    camera_location = camera['P_world'] @ location_hom
    camera_tip = camera['P_world'] @ tip_hom
    p1 = camera['K'] @ camera_location
    p2 = camera['K'] @ camera_tip
    p1 = (p1[:2] / p1[2]).flatten()
    p2 = (p2[:2] / p2[2]).flatten()
    # print(f"Projected points: p1={p1}, p2={p2}")
    
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    yaw_2d = np.arctan2(dy, dx)
    # print(f"2D vector (dx, dy): ({dx:.2f}, {dy:.2f}), 2D yaw: {yaw_2d:.6f} radians")
    return p1, p2, yaw_2d

def compute_h(h_pixel, depth_point, depth_frame, E, K):
    u = int(depth_point[0])
    v = int(depth_point[1])
    height, width = depth_frame.shape
    u = max(0, min(int(u), width - 1))  # Clamp x to [0, 1919]
    v = max(0, min(int(v), height - 1))  # Clamp y to [0, 1079]
    R_matrix = E[:, :3]
    rvec, _ = cv.Rodrigues(R_matrix)
    rotation = R.from_rotvec(rvec.flatten())
    _, pitch_cam, _ = rotation.as_euler('zyx', degrees=False)
    fy = K[1, 1]
    d  = depth_frame[v, u] / 1000.0
    h_estimate = (d * h_pixel) / fy
    # Convert to real-world height using camera pitch (in radians!)
    h_final = np.float16(h_estimate / np.sin(np.deg2rad(90) - pitch_cam) )

    return h_final

def compute_wl(pts: dict, bbox: list, class_id: int, depth_frame: np.ndarray, E: np.ndarray, K: np.ndarray, 
               thresh_shoulder: float = 0.1, thresh_ankle: float = 0.3,
               l_const: float = 0.5, w_const: float = 0.8):

    def to_world(coord):
        x_pix, y_pix = coord
        height, width = depth_frame.shape
        x_pix = max(0, min(int(x_pix), width - 1))  # Clamp x to [0, width-1]
        y_pix = max(0, min(int(y_pix), height - 1))  # Clamp y to [0, height-1]
        d = depth_frame[int(y_pix), int(x_pix)] / 1000
        if d <= 0:
            return None
        return project_pixel_to_3dworld(coord, d, E, K)

    # Initialize defaults
    w_final = w_const
    l_final = l_const

    if class_id == 0:  # Person
        # Shoulder
        l_sh_px = pts.get('L_Shoulder')
        r_sh_px = pts.get('R_Shoulder')
        if l_sh_px is not None and r_sh_px is not None:
            world_l_shoulder = to_world(l_sh_px)
            world_r_shoulder = to_world(r_sh_px)
        else:
            world_l_shoulder = world_r_shoulder = None

        # Ankle
        l_ankle_px = pts.get('L_Ankle')
        r_ankle_px = pts.get('R_Ankle')
        if l_ankle_px is not None and r_ankle_px is not None:
            world_l_ankle = to_world(l_ankle_px)
            world_r_ankle = to_world(r_ankle_px)
        else:
            world_l_ankle = world_r_ankle = None

        # Distance calculations
        shoulder_dist = 0.0
        if world_l_shoulder is not None and world_r_shoulder is not None:
            shoulder_dist = np.linalg.norm(world_l_shoulder - world_r_shoulder)

        ankle_dist = 0.0
        if world_l_ankle is not None and world_r_ankle is not None:
            ankle_dist = np.linalg.norm(world_l_ankle - world_r_ankle)

        is_shoulder_open = shoulder_dist >= thresh_shoulder
        is_ankle_open = ankle_dist >= thresh_ankle

        if is_shoulder_open:
            w_final = shoulder_dist
            if is_ankle_open:
                l_final = min(ankle_dist, 0.8)
            else:
                l_final = l_const
        else:
            w_final = w_const
            if is_ankle_open:
                l_final = min(ankle_dist, 0.8)
            else:
                l_final = l_const
    else:  # Non-person
        # Assume bbox is [x_min, y_min, x_max, y_max]
        if len(bbox) != 4:
            return np.float16( w_final), np.float16(l_final)  # Return defaults if bbox is invalid

        # Define corners of the bounding box
        top_left = (bbox[0], bbox[1])
        top_right = (bbox[2], bbox[1])
        bottom_left = (bbox[0], bbox[3])
        bottom_right = (bbox[2], bbox[3])

        # Project corners to 3D world coordinates
        world_top_left = to_world(top_left)
        world_top_right = to_world(top_right)
        world_bottom_left = to_world(bottom_left)
        world_bottom_right = to_world(bottom_right)

        # Calculate width (distance between left and right edges)
        if world_top_left is not None and world_top_right is not None:
            w_final = np.linalg.norm(world_top_left - world_top_right)
        elif world_bottom_left is not None and world_bottom_right is not None:
            w_final = np.linalg.norm(world_bottom_left - world_bottom_right)
        else:
            w_final = w_const

        # Calculate length (distance between top and bottom edges)
        if world_top_left is not None and world_bottom_left is not None:
            l_final = np.linalg.norm(world_top_left - world_bottom_left)
        elif world_top_right is not None and world_bottom_right is not None:
            l_final = np.linalg.norm(world_top_right - world_bottom_right)
        else:
            l_final = l_const

        # Apply constraints
        w_final = min(w_final, 0.8)
        l_final = min(l_final, 0.8)

    return np.float16(w_final), np.float16(l_final)

def compute_yaw(pts, depth_frame, E, K, bbox_center):
    """
    Compute yaw from shoulder first, 
    if dont have shoulder or person is rotating then use all the point in face to calculate the yaw
    """
    def to_word(coord):
        x_pix, y_pix = coord
        height, width = depth_frame.shape
        x_pix = max(0, min(int(x_pix), width - 1))  # Clamp x to [0, 1919]
        y_pix = max(0, min(int(y_pix), height - 1))  # Clamp y to [0, 1079]
        d = depth_frame[int(y_pix), int(x_pix)] / 1000
        if d <= 0:
            return None
        return project_pixel_to_3dworld(coord, d, E, K)
    
    ls = pts.get('L_Shoulder')
    rs = pts.get('R_Shoulder')
    world_bbox = to_word(bbox_center)

    if (ls is not None) and (rs is not None):
        world_l_shoulder = to_word(ls)
        world_r_shoulder = to_word(rs)

        dx_sh = world_l_shoulder[0] - world_r_shoulder[0]
        dy_sh = world_l_shoulder[1] - world_r_shoulder[1]
        if math.hypot(dx_sh, dy_sh) >= 1e-3:
            yaw = math.atan2(dy_sh, dx_sh)
            return yaw, world_bbox

    # fallback: centroid all keypoint in face
    face_keys = ['Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear']
    bev_face_pts = []
    for k in face_keys:
        kp = pts.get(k)
        if kp is not None:
            world_face = to_word(kp)
            x_face = world_face[0]
            y_face = world_face[1]
            bev_face_pts.append((x_face, y_face))
    
        if bev_face_pts:
            arr = np.array(bev_face_pts, dtype=float)
            x_face = float(arr[:, 0].mean())
            y_face = float(arr[:, 1].mean())
            # print(bbox_center)
            # project center bbox 2d into bev
            if world_bbox is not None:
                x_box, y_box = world_bbox[0], world_bbox[1]  

                dx_f = x_face - x_box
                dy_f = y_face - y_box
                if math.hypot(dx_f, dy_f) >= 1e-3:
                    yaw = math.atan2(dy_f, dx_f)
                    return yaw, world_bbox
        
    # if not shoulder either face then fallback yaw = 0
    return 0.0, world_bbox


def compute_yaw3d_from_yaw2d(pixel_center, yaw_2d, depth_frame, E, K):
    """
    Compute 3D yaw from 2D yaw using depth frame and camera parameters.
    
    Args:
        pixel_center (tuple or np.ndarray): 2D pixel coordinates (u, v) of the object center.
        yaw_2d (float): 2D yaw angle in radians (computed in the image plane).
        depth_frame (np.ndarray): Depth frame (in millimeters) with shape (height, width).
        E (np.ndarray): Extrinsic matrix (4x4 or 3x4) for camera-to-world transformation.
        K (np.ndarray): Intrinsic matrix (3x3) of the camera.
    
    Returns:
        float: 3D yaw angle in radians, or None if computation fails (e.g., invalid depth).
    """
    # Ensure inputs are numpy arrays
    pixel_center = np.array(pixel_center, dtype=float)
    height, width = depth_frame.shape
    
    # Clamp pixel coordinates to valid image bounds
    u, v = pixel_center
    u = max(0, min(int(u), width - 1))
    v = max(0, min(int(v), height - 1))
    
    # Get depth value at the pixel (convert from mm to meters)
    d = depth_frame[v, u] / 1000.0
    if d <= 0:
        print("Invalid depth value at pixel ({}, {}).".format(u, v))
        return None
    
    # Project the center pixel to 3D world coordinates
    world_center = project_pixel_to_3dworld(pixel_center, d, E, K)
    
    # Compute a second point in the image plane along the 2D yaw direction
    # Assume a small offset (e.g., 10 pixels) along the yaw direction
    offset = 10.0  # Pixel offset to define direction
    u2 = u + offset * np.cos(yaw_2d)
    v2 = v + offset * np.sin(yaw_2d)
    
    # Clamp the second point to image bounds
    u2 = max(0, min(int(u2), width - 1))
    v2 = max(0, min(int(v2), height - 1))
    
    # Get depth value for the second point
    d2 = depth_frame[v2, u2] / 1000.0
    if d2 <= 0:
        print("Invalid depth value at second pixel ({}, {}).".format(u2, v2))
        return None
    
    # Project the second pixel to 3D world coordinates
    world_tip = project_pixel_to_3dworld([u2, v2], d2, E, K)
    
    # Compute the 3D direction vector
    direction = world_tip - world_center
    
    # Compute yaw in the world coordinate system (in the XY plane)
    dx = direction[0]
    dy = direction[1]
    if np.hypot(dx, dy) < 1e-6:
        print("Direction vector too small to compute yaw.")
        return None
    
    yaw_3d = np.arctan2(dy, dx)
    
    return yaw_3d

def get_center_obj(detection):
    
    x1, y1, x2, y2, _ = detection.bbox

    if detection.class_id ==0:
        if detection.keypoints_2d[11][-1] >0.5 and detection.keypoints_2d[12][-1] >0.5 :
            bbox_center = np.array([
                (detection.keypoints_2d[11][0] + detection.keypoints_2d[12][0]) / 2, 
                (detection.keypoints_2d[11][1] + detection.keypoints_2d[12][1]) / 2 
            ])
        else:
            bbox_center = np.array([
                (x1+ x2) / 2, 
                (y1 + y2) / 2 +10
            ])
    elif detection.class_id == 2 or detection.class_id == 3:
        bbox_center = np.array([
                (x1+ x2) / 2, 
                (y1 + y2) / 2  
        ])
    
    else:
        if detection.keypoints_2d[11][-1] >0.5 and detection.keypoints_2d[12][-1] >0.5 :
            bbox_center = np.array([
                (detection.keypoints_2d[11][0] + detection.keypoints_2d[12][0]) / 2, 
                (detection.keypoints_2d[11][1] + detection.keypoints_2d[12][1]) / 2 
            ])
        else:
            bbox_center = np.mean( [detection.keypoints_2d[i][:2] for i in [5,6,11,12]], axis=0)
    return bbox_center


def discrete_frechet_distance(traj1, traj2):
    """
    Compute the discrete Fréchet distance between two 3D trajectories in a homography plane.
    Args:
        traj1: numpy array of shape (n, 3), where n is the number of points (x, y, z).
        traj2: numpy array of shape (m, 3), where m is the number of points (x, y, z).
    Returns:
        float: The discrete Fréchet distance.
    """
    n, m = len(traj1), len(traj2)
    if n == 0 or m == 0:
        return np.inf

    # Initialize the cost matrix
    ca = np.full((n, m), np.inf)
    
    # Compute 3D Euclidean distance
    def euclidean_dist(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    # Base case: distance between first points
    ca[0, 0] = euclidean_dist(traj1[0], traj2[0])
    
    # Fill first row and column
    for i in range(1, n):
        ca[i, 0] = max(ca[i-1, 0], euclidean_dist(traj1[i], traj2[0]))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j-1], euclidean_dist(traj1[0], traj2[j]))
    
    # Fill the rest of the cost matrix
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(
                min(ca[i-1, j], ca[i, j-1], ca[i-1, j-1]),
                euclidean_dist(traj1[i], traj2[j])
            )
    
    return ca[n-1, m-1]



def compute_3d_iou_fast_approximation(bbox1, bbox2, yaw_threshold=np.pi/3, yaw_penalty_factor=2.0):
    """
    Fast approximation of 3D IoU using axis-aligned bounding boxes, with yaw difference penalty.
    
    Args:
        bbox1 (dict): {'center': np.array([x, y, z]), 'dims': np.array([w, h, l]), 'yaw': float}
        bbox2 (dict): {'center': np.array([x, y, z]), 'dims': np.array([w, h, l]), 'yaw': float}
        yaw_threshold (float): Yaw difference threshold in radians (default: 60 degrees = pi/3)
        yaw_penalty_factor (float): Controls the strength of the yaw penalty (higher = stronger penalty)
    
    Returns:
        float: IoU value adjusted for yaw difference
    """
    # Compute yaw difference
    yaw1 = bbox1['yaw'] % (2 * np.pi)  # Normalize to [0, 2π]
    yaw2 = bbox2['yaw'] % (2 * np.pi)
    yaw_diff = min(abs(yaw1 - yaw2), 2 * np.pi - abs(yaw1 - yaw2))  # Smallest angular difference
    
    # Compute base 3D IoU
    R1 = rotation_matrix(bbox1['yaw'])
    R2 = rotation_matrix(bbox2['yaw'])
    
    corners1 = np.array(create_corners(bbox1['dims'], location=bbox1['center'], R=R1))
    corners2 = np.array(create_corners(bbox2['dims'], location=bbox2['center'], R=R2))
    
    # Find axis-aligned bounding boxes
    min1, max1 = np.min(corners1, axis=0), np.max(corners1, axis=0)
    min2, max2 = np.min(corners2, axis=0), np.max(corners2, axis=0)
    
    # Compute intersection
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_dims = np.maximum(inter_max - inter_min, 0)
    intersection = np.prod(inter_dims) if np.all(inter_max > inter_min) else 0
    
    # Compute union
    vol1 = np.prod(bbox1['dims'])
    vol2 = np.prod(bbox2['dims'])
    union = vol1 + vol2 - intersection
    
    base_iou = intersection / union if union > 0 else 0
    
    # Apply yaw penalty
    if yaw_diff <= yaw_threshold:
        penalty = 1.0  # No penalty for small yaw differences
    else:
        # Exponential penalty: exp(-k * (yaw_diff - threshold))
        penalty = np.exp(-yaw_penalty_factor * (yaw_diff - yaw_threshold))
        # Cap penalty to avoid reducing IoU to near zero
        penalty = max(penalty, 0.1)
    
    return base_iou * penalty
    # return base_iou
def compute_3d_iou_convex_hull(bbox1, bbox2, epsilon=1e-6, verbose=False):
    """
    Compute the 3D IoU of two oriented 3D bounding boxes using convex hull method.
    
    Args:
        bbox1 (dict): Dictionary with 'center' (3D numpy array), 'dims' (width, length, height), and 'yaw' (float).
        bbox2 (dict): Same as bbox1 for the second box.
        epsilon (float): Small value to avoid division by zero or numerical instability.
        verbose (bool): If True, print debug information.
    
    Returns:
        float: 3D IoU value, or 0 if boxes do not intersect or union is zero.
    """
    # Input validation
    if not all(key in bbox1 for key in ['center', 'dims', 'yaw']) or \
       not all(key in bbox2 for key in ['center', 'dims', 'yaw']):
        return 0.0
    if np.any(np.array(bbox1['dims']) <= 0) or np.any(np.array(bbox2['dims']) <= 0) or \
       np.any(np.isnan(bbox1['center'])) or np.any(np.isnan(bbox2['center'])) or \
       np.isnan(bbox1['yaw']) or np.isnan(bbox2['yaw']):
        return 0.0

    # Generate corners for both bounding boxes
    R1 = rotation_matrix(bbox1['yaw'])
    R2 = rotation_matrix(bbox2['yaw'])
    corners1 = np.array(create_corners(bbox1['dims'], location=bbox1['center'], R=R1))
    corners2 = np.array(create_corners(bbox2['dims'], location=bbox2['center'], R=R2))

    # Compute volumes of individual boxes
    vol1 = np.prod(bbox1['dims'])
    vol2 = np.prod(bbox2['dims'])

    # Quick axis-aligned check to reject non-overlapping boxes
    min1, max1 = np.min(corners1, axis=0), np.max(corners1, axis=0)
    min2, max2 = np.min(corners2, axis=0), np.max(corners2, axis=0)
    if not np.all(max1 >= min2) or not np.all(max2 >= min1):
        return 0.0

    # Compute intersection points
    intersection_points = []

    # Check which corners of bbox1 are inside bbox2 and vice versa
    for corner in corners1:
        if is_point_inside_box(corner, corners2, bbox2['center'], bbox2['dims'], R2):
            intersection_points.append(corner)
    for corner in corners2:
        if is_point_inside_box(corner, corners1, bbox1['center'], bbox1['dims'], R1):
            intersection_points.append(corner)

    # Define edges
    edges1 = [(corners1[i], corners1[j]) for i, j in [
        (0,1), (1,2), (2,3), (3,0),  # Top face
        (4,5), (5,6), (6,7), (7,4),  # Bottom face
        (0,4), (1,5), (2,6), (3,7)   # Vertical edges
    ]]
    edges2 = [(corners2[i], corners2[j]) for i, j in [
        (0,1), (1,2), (2,3), (3,0),  # Top face
        (4,5), (5,6), (6,7), (7,4),  # Bottom face
        (0,4), (1,5), (2,6), (3,7)   # Vertical edges
    ]]

    # Compute edge-plane intersection points
    intersection_points.extend(compute_edge_face_intersections(edges1, corners2, R2, bbox2['center'], bbox2['dims']))
    intersection_points.extend(compute_edge_face_intersections(edges2, corners1, R1, bbox1['center'], bbox1['dims']))

    # Remove duplicates with higher tolerance and filter invalid points
    if intersection_points:
        intersection_points = np.array(intersection_points)
        # Remove NaN or infinite points
        valid_mask = np.all(np.isfinite(intersection_points), axis=1)
        intersection_points = intersection_points[valid_mask]
        if len(intersection_points) > 0:
            # Remove duplicates with tolerance
            intersection_points = np.unique(np.round(intersection_points, decimals=10), axis=0)

    # Compute intersection volume
    intersection_vol = 0.0
    if len(intersection_points) >= 4:
        try:
            # Check if points span 3D space
            points_centered = intersection_points - np.mean(intersection_points, axis=0)
            rank = np.linalg.matrix_rank(points_centered, tol=1e-10)
            if rank < 3:
                if verbose:
                    print("Points are coplanar, falling back to fast approximation.")
                return compute_3d_iou_fast_approximation(bbox1, bbox2)
            
            # Use QJ to joggle points
            hull = ConvexHull(intersection_points, qhull_options='QJ')
            intersection_vol = hull.volume
        except Exception as e:
            if verbose:
                print(f"ConvexHull failed: {e}. Falling back to fast approximation.")
            return compute_3d_iou_fast_approximation(bbox1, bbox2)
    elif len(intersection_points) > 0 and verbose:
        print(f"Insufficient points ({len(intersection_points)} < 4), falling back to fast approximation.")
        return compute_3d_iou_fast_approximation(bbox1, bbox2)

    # Compute union
    union_vol = vol1 + vol2 - intersection_vol

    if verbose:
        print(f"Intersection volume: {intersection_vol}, Union volume: {union_vol}")

    # Return IoU
    return intersection_vol / (union_vol + epsilon) if union_vol > 0 else 0.0

def is_point_inside_box(point, box_corners, box_center, box_dims, R, epsilon=1e-6):
    """
    Check if a point is inside an oriented 3D bounding box.
    
    Args:
        point (np.ndarray): 3D point to check.
        box_corners (np.ndarray): Corners of the box.
        box_center (np.ndarray): Center of the box.
        box_dims (np.ndarray): Dimensions (width, length, height).
        R (np.ndarray): Rotation matrix.
        epsilon (float): Tolerance for numerical stability.
    
    Returns:
        bool: True if point is inside the box.
    """
    # Transform point to box's local coordinate system
    point_local = np.dot(R.T, (point - box_center))
    
    # Check if point lies within the box's dimensions
    half_dims = np.array(box_dims) / 2
    return np.all(np.abs(point_local) <= half_dims + epsilon)

def compute_edge_face_intersections(edges, box_corners, R, box_center, box_dims):
    """
    Compute intersection points between edges of one box and faces of another.
    
    Args:
        edges (list): List of edge tuples (start, end) from one box.
        box_corners (np.ndarray): Corners of the other box.
        R (np.ndarray): Rotation matrix of the other box.
        box_center (np.ndarray): Center of the other box.
        box_dims (np.ndarray): Dimensions of the other box.
    
    Returns:
        list: Intersection points.
    """
    intersection_points = []
    # Define the 6 faces of the box (corrected)
    faces = [
        [0,1,2,3],  # Top face (z = dz)
        [4,5,6,7],  # Bottom face (z = -dz)
        [0,1,5,4],  # Front face (y = -dy)
        [2,3,7,6],  # Back face (y = dy)
        [1,2,6,5],  # Left face (x = -dx)
        [0,3,7,4]   # Right face (x = dx)
    ]
    
    # Transform box to local coordinates
    box_corners_local = np.dot(R.T, (box_corners - box_center).T).T
    half_dims = np.array(box_dims) / 2

    for edge in edges:
        p1, p2 = np.array(edge[0]), np.array(edge[1])
        # Transform edge points to the other box's local coordinates
        p1_local = np.dot(R.T, (p1 - box_center))
        p2_local = np.dot(R.T, (p2 - box_center))
        
        # Check intersection with each face
        for face_idx in faces:
            # Define face bounds in local coordinates
            face_corners = box_corners_local[face_idx]
            min_bound = np.min(face_corners, axis=0)
            max_bound = np.max(face_corners, axis=0)
            
            # Check axis-aligned intersection in local coordinates
            t = (min_bound - p1_local) / (p2_local - p1_local + 1e-10)
            t = t[~np.isnan(t) & (t >= 0) & (t <= 1)]
            
            for t_val in t:
                intersect_point = p1_local + t_val * (p2_local - p1_local)
                # Check if intersection point lies within face bounds
                if np.all(intersect_point >= min_bound - 1e-6) and np.all(intersect_point <= max_bound + 1e-6):
                    # Transform back to world coordinates
                    world_point = np.dot(R, intersect_point) + box_center
                    # Clip to box bounds to avoid numerical errors
                    local_check = np.abs(intersect_point) <= half_dims + 1e-6
                    if np.all(local_check):
                        intersection_points.append(world_point)
    
    return intersection_points


def segment_object_by_depth(bbox, depth_frame, depth_threshold=1000):
    """
    Segment object pixels within the 2D bbox based on depth continuity.
    
    Args:
        bbox (list): [x1, y1, x2, y2, confidence]
        depth_frame (np.ndarray): Depth frame in millimeters
        depth_threshold (float): Maximum depth difference for segmentation (mm)
    
    Returns:
        np.ndarray: Binary mask of segmented object pixels
    """
    x1, y1, x2, y2 = map(int, bbox[:4])
    height, width = depth_frame.shape
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    depth_crop = depth_frame[y1:y2, x1:x2]
    if depth_crop.size == 0:
        return None

    valid_depths = depth_crop[depth_crop > 0]
    if valid_depths.size == 0:
        return None
    median_depth = np.median(valid_depths)

    mask = np.zeros_like(depth_crop, dtype=np.uint8)
    mask[(depth_crop > 0) & (np.abs(depth_crop - median_depth) < depth_threshold)] = 255

    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask

    return full_mask

def compute_3d_center_from_segmentation(bbox, depth_frame, E, K, depth_threshold=1000):
    """
    Compute the 3D bbox center by segmenting the object and averaging 3D points.
    
    Args:
        bbox (list): [x1, y1, x2, y2, confidence]
        depth_frame (np.ndarray): Depth frame in millimeters
        E (np.ndarray): Extrinsic matrix
        K (np.ndarray): Intrinsic matrix
        depth_threshold (float): Maximum depth difference for segmentation (mm)
    
    Returns:
        np.ndarray: 3D center point [x, y, z], or None if computation fails
    """
    mask = segment_object_by_depth(bbox, depth_frame, depth_threshold)
    if mask is None:
        return None

    height, width = depth_frame.shape
    y, x = np.where(mask == 255)
    if len(x) == 0:
        return None

    pixels = np.stack([x, y], axis=1)  # [N, 2] array of [u, v]
    depths = depth_frame[y, x] / 1000.0
    valid = depths > 0
    if not np.any(valid):
        return None

    pixels = pixels[valid]
    depths = depths[valid]

    points_3d = np.zeros((len(pixels), 3))
    for i, (u, v) in enumerate(pixels):
        points_3d[i] = project_pixel_to_3dworld([u, v], depths[i], E, K)

    center_3d = np.mean(points_3d, axis=0)
    return center_3d

def compute_yaw_from_min_area_rect(mask):
    """
    Compute 2D yaw using cv2.minAreaRect on segmented object pixels.
    
    Args:
        mask (np.ndarray): Binary mask of segmented object
    
    Returns:
        float: 2D yaw angle in radians, or None if computation fails
    """
    if mask is None:
        return None

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv.contourArea)
    if cv.contourArea(largest_contour) < 50:
        return None

    rect = cv.minAreaRect(largest_contour)
    (cx, cy), (w, h), angle = rect

    yaw = np.deg2rad(angle)
    if w < h:
        yaw += np.pi / 2

    yaw = yaw % (2 * np.pi)
    return yaw

def compute_yaw_from_3d_points(bbox, depth_frame, E, K, depth_threshold=1000):
    """
    Compute 3D yaw using PCA on the 3D point cloud of the segmented object.
    
    Args:
        bbox (list): [x1, y1, x2, y2, confidence]
        depth_frame (np.ndarray): Depth frame in millimeters
        E (np.ndarray): Extrinsic matrix
        K (np.ndarray): Intrinsic matrix
        depth_threshold (float): Maximum depth difference for segmentation (mm)
    
    Returns:
        float: 3D yaw angle in radians, or None if computation fails
    """
    mask = segment_object_by_depth(bbox, depth_frame, depth_threshold)
    if mask is None:
        return None

    height, width = depth_frame.shape
    y, x = np.where(mask == 255)
    if len(x) < 10:  # Minimum number of points
        return None

    pixels = np.stack([x, y], axis=1)
    depths = depth_frame[y, x] / 1000.0
    valid = depths > 0
    if not np.any(valid):
        return None

    pixels = pixels[valid]
    depths = depths[valid]

    points_3d = np.zeros((len(pixels), 3))
    for i, (u, v) in enumerate(pixels):
        points_3d[i] = project_pixel_to_3dworld([u, v], depths[i], E, K)

    # Project to ground plane (z=0) for yaw estimation
    points_2d = points_3d[:, :2]  # Use x, y coordinates

    # Apply PCA
    pca = PCA(n_components=2)
    pca.fit(points_2d)
    principal_axis = pca.components_[0]  # First principal component

    # Compute yaw (angle of principal axis in XY plane)
    yaw = np.arctan2(principal_axis[1], principal_axis[0])
    yaw = yaw % (2 * np.pi)

    return yaw

def project_3dcamera_to_2d(point_3d, intrinsics):
    X, Y, Z = point_3d
    if abs(Z) < 1e-6:
        return None
    u = (intrinsics['fx'] * X / Z) + intrinsics['cx']
    v = (intrinsics['fy'] * Y / Z) + intrinsics['cy']
    return int(u), int(v)

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

def calculate_3d_yaw_white_humanoid(mask, cropped_depth, camera_intrinsics, x1, y1):
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics["fy"]
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']

    v_mask, u_mask = np.where(mask > 0)
    if len(v_mask) < 20:
        return (None,)*7
    
    min_v, max_v = np.min(v_mask), np.max(v_mask)
    height = max_v - min_v
    if height == 0:
        return None, None, None, None, None, None, None
    
    upper_body_v_start = min_v
    upper_body_v_end = min_v + int(height * 0.5)

    upper_body_mask = np.zeros_like(mask)
    upper_body_mask[upper_body_v_start : upper_body_v_end, :] = mask[upper_body_v_start : upper_body_v_end, :]

    v, u = np.where(upper_body_mask > 0)
    d = cropped_depth[v, u]
    valid_indices = d > 0
    u, v, d = u[valid_indices], v[valid_indices], d[valid_indices]

    if len(u) < 20:
        return (None,)*7
    
    crop_h, crop_w = cropped_depth.shape
    cx_crop = cx - x1
    cy_crop = cy - y1

    x = (u - cx_crop) * d /fx
    y = (v - cy_crop) * d /fy
    z = d

    points_3d = np.vstack((x, y, z)).T

    mean_z = np.mean(z)
    std_z = np.std(z)
    z_threshold = 2.0
    z_mask = (z >= mean_z - z_threshold * std_z) & (z <= mean_z + z_threshold * std_z)
    points_3d = points_3d[z_mask]

    if points_3d.shape[0] < 50:
        return None, None, None

    centroid = np.mean(points_3d, axis=0)
    distances = np.linalg.norm(points_3d - centroid, axis=1)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    dist_threshold = 2.0
    dist_mask = distances <= mean_dist + dist_threshold * std_dist
    points_3d = points_3d[dist_mask]

    if points_3d.shape[0] < 20:
        return (None,)*7
    
    mean, eigenvectors, _ = cv.PCACompute2(np.float32(points_3d), mean=np.empty((0)))
    normal_vector = eigenvectors[2]
    centroid_3d = mean[0]

    if np.dot(normal_vector, centroid_3d) > 0:
        normal_vector = -normal_vector

    k = 250.0
    P1 = centroid_3d
    P2 = centroid_3d + k * normal_vector

    intrinsics_crop = {'fx': fx, 'fy': fy, 'cx': cx_crop, 'cy': cy_crop}
    start_point_2d_crop = project_3dcamera_to_2d(P1, intrinsics_crop)
    end_point_2d_crop = project_3dcamera_to_2d(P2, intrinsics_crop)

    intrinsics_full = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
    start_point_2d_full = project_3dcamera_to_2d(P1, intrinsics_full)
    end_point_2d_full = project_3dcamera_to_2d(P2, intrinsics_full)

    if start_point_2d_crop is None or end_point_2d_crop is None or \
       start_point_2d_full is None or end_point_2d_full is None:
        return (None,)*7

    return start_point_2d_crop, end_point_2d_crop, start_point_2d_full, end_point_2d_full, points_3d, normal_vector, centroid_3d