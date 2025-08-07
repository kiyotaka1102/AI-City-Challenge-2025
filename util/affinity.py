import numpy as np

# Constants
EPSILON = 1e-6 
NUM_KEYPOINTS = 17  
FEET_IDXS = [15, 16]  

def validate_array_shape(arr, expected_shape, name):
    """Validate the shape of a NumPy array."""
    if arr.ndim != len(expected_shape):
        raise ValueError(f"{name} has incorrect number of dimensions: expected {len(expected_shape)}, got {arr.ndim}")
    for i, (actual, expected) in enumerate(zip(arr.shape, expected_shape)):
        if expected >= 0 and actual != expected:
            raise ValueError(f"{name} has incorrect shape at dimension {i}: expected {expected}, got {actual}")

def epipolar_3d_score_norm(pA, rayA, pB, rayB, alpha_epi):
    """
    Compute epipolar geometry score between two camera views.
    
    Args:
        pA (np.ndarray): Camera A position, shape (3,)
        rayA (np.ndarray): Rays from camera A, shape (num_keypoints, 3)
        pB (np.ndarray): Camera B position, shape (3,)
        rayB (np.ndarray): Rays from camera B, shape (num_keypoints, 3)
        alpha_epi (float): Epipolar threshold
    
    Returns:
        np.ndarray: Epipolar scores, shape (num_keypoints,)
    """
    validate_array_shape(pA, [3], "pA")
    validate_array_shape(pB, [3], "pB")
    validate_array_shape(rayA, [-1, 3], "rayA")  
    validate_array_shape(rayB, [rayA.shape[0], 3], "rayB")  

    # Cross product: rayA x rayB
    cp = np.cross(rayA, rayB, axis=1)  
    norm = np.sqrt(np.sum(cp**2, axis=1)) + EPSILON  
    
    # Dot product: (pA - pB) * cp
    p_diff = pA - pB  # Shape: (3,)
    dot_product = np.sum(p_diff * cp, axis=1) 
    
    # Epipolar distance
    dist = np.abs(dot_product) / norm
    return 1.0 - dist / alpha_epi

def compute_joints_rays(keypoints_mv, cam_project_inv, cam_pos):
    """
    Compute normalized ray directions for keypoints.
    
    Args:
        keypoints_mv (np.ndarray): Keypoints, shape (num_keypoints, 3)
        cam_project_inv (np.ndarray): Inverse projection matrix, shape (4, 3)
        cam_pos (np.ndarray): Camera position, shape (3,)
    
    Returns:
        np.ndarray: Normalized joint rays, shape (num_keypoints, 3)
    """
    validate_array_shape(keypoints_mv, [NUM_KEYPOINTS, 3], "keypoints_mv")
    validate_array_shape(cam_project_inv, [4, 3], "cam_project_inv")
    validate_array_shape(cam_pos, [3], "cam_pos")

    # Homogeneous keypoints: (x, y, 1)
    joints_h = np.vstack((keypoints_mv[:, :2].T, np.ones((1, NUM_KEYPOINTS))))  
    
    # Project to world coordinates
    joints_rays = cam_project_inv @ joints_h  
    joints_rays /= joints_rays[3, :] + EPSILON  
    joints_rays = joints_rays[:3, :] - cam_pos[:, None] 
    
    # Normalize rays
    norm = np.sqrt(np.sum(joints_rays**2, axis=0)) + EPSILON
    joints_rays_norm = (joints_rays / norm).T  
    return joints_rays_norm

def aff_sum(aff_temp, valid, age_2D_sv, age_2D_thr):
    """
    Compute weighted sum of affinity scores.
    
    Args:
        aff_temp (np.ndarray): Affinity scores, shape (num_keypoints,)
        valid (np.ndarray): Valid keypoint mask, shape (num_keypoints,)
        age_2D_sv (np.ndarray): Age of 2D keypoints, shape (num_keypoints,)
        age_2D_thr (float): Age threshold
    
    Returns:
        float: Weighted affinity sum
    """
    validate_array_shape(aff_temp, [NUM_KEYPOINTS], "aff_temp")
    validate_array_shape(valid, [NUM_KEYPOINTS], "valid")
    validate_array_shape(age_2D_sv, [NUM_KEYPOINTS], "age_2D_sv")

    weights = valid * np.exp(-age_2D_sv) * (age_2D_sv <= age_2D_thr)
    aff = np.sum(aff_temp * weights)
    aff_norm = np.sum(weights) + EPSILON
    return aff / aff_norm

def compute_feet_distance(joints, feet_idxs, homo_feet_inv, feet_s, thred_homo):
    """
    Compute distance between feet positions in world coordinates.
    
    Args:
        joints (np.ndarray): Keypoints, shape (num_keypoints, 3)
        feet_idxs (list): Indices of feet keypoints
        homo_feet_inv (np.ndarray): Inverse feet homography, shape (3, 3)
        feet_s (np.ndarray): Source feet position, shape (2,)
        thred_homo (float): Homography threshold
    
    Returns:
        float: Normalized distance
    """
    validate_array_shape(joints, [NUM_KEYPOINTS, 3], "joints")
    validate_array_shape(homo_feet_inv, [3, 3], "homo_feet_inv")
    validate_array_shape(feet_s, [2], "feet_s")
    if not feet_idxs:
        raise ValueError("feet_idxs cannot be empty")
    for i in feet_idxs:
        if i < 0 or i >= NUM_KEYPOINTS:
            raise ValueError(f"Invalid feet_idx: {i}")

    feet_pos = np.mean(joints[feet_idxs, :2], axis=0)  # Shape: (2,)
    feet_t_h = np.array([feet_pos[0], feet_pos[1], 1.0])  # Homogeneous coordinates
    feet_t = homo_feet_inv @ feet_t_h  # Shape: (3,)
    feet_t = feet_t[:2] / (feet_t[2] + EPSILON)  # Normalize to 2D
    norm = np.sqrt(np.sum((feet_s - feet_t)**2))
    return 1.0 - norm / thred_homo

def compute_feet_s(joints, feet_idxs, homo_feet_inv):
    """
    Compute world coordinates of feet keypoints.
    
    Args:
        joints (np.ndarray): Keypoints, shape (num_keypoints, 3)
        feet_idxs (list): Indices of feet keypoints
        homo_feet_inv (np.ndarray): Inverse feet homography, shape (3, 3)
    
    Returns:
        np.ndarray: Feet position in world coordinates, shape (2,)
    """
    validate_array_shape(joints, [NUM_KEYPOINTS, 3], "joints")
    validate_array_shape(homo_feet_inv, [3, 3], "homo_feet_inv")
    if not feet_idxs:
        raise ValueError("feet_idxs cannot be empty")
    for i in feet_idxs:
        if i < 0 or i >= NUM_KEYPOINTS:
            raise ValueError(f"Invalid feet_idx: {i}")

    feet_pos = np.mean(joints[feet_idxs, :2], axis=0)  # Shape: (2,)
    feet_t_h = np.array([feet_pos[0], feet_pos[1], 1.0])  # Homogeneous coordinates
    feet_t = homo_feet_inv @ feet_t_h  # Shape: (3,)
    return feet_t[:2] / (feet_t[2] + EPSILON)  # Shape: (2,)

def compute_box_pos_s(box, homo_inv):
    """
    Compute world coordinates of bounding box bottom center.
    
    Args:
        box (np.ndarray): Bounding box [x1, y1, x2, y2, conf], shape (5,)
        homo_inv (np.ndarray): Inverse homography, shape (3, 3)
    
    Returns:
        np.ndarray: Box position in world coordinates, shape (2,)
    """
    validate_array_shape(box, [5], "box")
    validate_array_shape(homo_inv, [3, 3], "homo_inv")

    box_pos = np.array([(box[0] + box[2]) / 2.0, box[3], 1.0])  # Bottom center, homogeneous
    box_pos_w_h = homo_inv @ box_pos  # Shape: (3,)
    return box_pos_w_h[:2] / (box_pos_w_h[2] + EPSILON)  # Shape: (2,)

def loop_t_homo_full(joints_t, joints_s, age_bbox, age_2D, feet_s, feet_valid_s, v, thred_epi, thred_homo,
                     keypoint_thrd, age_2D_thr, sv_ray, cameras, bbox_s, box_valid_s, bbox_mv_t):
    """
    Compute epipolar and homography-based affinity scores.
    
    Args:
        joints_t (np.ndarray): Track keypoints, shape (num_cams, num_keypoints, 3)
        joints_s (np.ndarray): Detection keypoints, shape (num_keypoints, 3)
        age_bbox (np.ndarray): Bounding box ages, shape (num_cams,)
        age_2D (np.ndarray): Keypoint ages, shape (num_cams, num_keypoints)
        feet_s (np.ndarray): Source feet position, shape (2,)
        feet_valid_s (bool): Whether source feet are valid
        v (int): Current camera index
        thred_epi (float): Epipolar threshold
        thred_homo (float): Homography threshold
        keypoint_thrd (float): Keypoint confidence threshold
        age_2D_thr (float): Age threshold for keypoints
        sv_ray (np.ndarray): Rays for detection, shape (num_keypoints, 3)
        cameras (list): List of camera objects
        bbox_s (np.ndarray): Source bounding box position, shape (2,)
        box_valid_s (bool): Whether source box is valid
        bbox_mv_t (np.ndarray): Track bounding boxes, shape (num_cams, 5)
    
    Returns:
        list: [epipolar_affinity, homography_affinity]
    """
    validate_array_shape(joints_t, [-1, NUM_KEYPOINTS, 3], "joints_t")
    validate_array_shape(joints_s, [NUM_KEYPOINTS, 3], "joints_s")
    validate_array_shape(age_bbox, [-1], "age_bbox")
    validate_array_shape(age_2D, [-1, NUM_KEYPOINTS], "age_2D")
    validate_array_shape(feet_s, [2], "feet_s")
    validate_array_shape(sv_ray, [NUM_KEYPOINTS, 3], "sv_ray")
    validate_array_shape(bbox_s, [2], "bbox_s")
    validate_array_shape(bbox_mv_t, [-1, 5], "bbox_mv_t")

    num_cams = len(cameras)
    if joints_t.shape[0] != num_cams or age_bbox.shape[0] != num_cams or age_2D.shape[0] != num_cams:
        raise ValueError("Camera-related arrays must have consistent number of cameras")

    pos = cameras[v].pos  # Shape: (3,)
    aff_ss_sum = 0.0
    aff_ss_cnt = EPSILON
    aff_homo_ss_sum = 0.0
    aff_homo_ss_cnt = EPSILON

    for vj in range(num_cams):
        if v == vj or age_bbox[vj] >= 2:
            continue

        cam_pos = cameras[vj].pos  # Shape: (3,)
        cam_project_inv = cameras[vj].project_inv  # Shape: (4, 3)

        # Compute track rays for view vj
        track_rays = compute_joints_rays(joints_t[vj], cam_project_inv, cam_pos)  # Shape: (num_keypoints, 3)
        
        # Epipolar score
        valid = (joints_t[vj, :, 2] > keypoint_thrd) & (joints_s[:, 2] > keypoint_thrd) & (age_2D[vj] <= age_2D_thr)
        aff_temp = epipolar_3d_score_norm(pos, sv_ray, cam_pos, track_rays, thred_epi)
        aff = aff_sum(aff_temp, valid, age_2D[vj], age_2D_thr)
        if aff != 0:
            aff_ss_sum += aff
            aff_ss_cnt += 1

        # Homography score
        feet_valid_t = np.all(joints_t[vj, FEET_IDXS, 2] > keypoint_thrd)
        if feet_valid_s and feet_valid_t:
            homo_feet_inv = cameras[vj].homo_feet_inv  # Shape: (3, 3)
            aff_homo = compute_feet_distance(joints_t[vj], FEET_IDXS, homo_feet_inv, feet_s, thred_homo)
            aff_homo_ss_sum += aff_homo
            aff_homo_ss_cnt += 1
        else:
            homo_inv = cameras[vj].homo_inv  # Shape: (3, 3)
            if bbox_mv_t[vj, 3] >= 1075 or not box_valid_s:
                continue
            box_pos_t = compute_box_pos_s(bbox_mv_t[vj], homo_inv)
            norm = np.sqrt(np.sum((box_pos_t - bbox_s)**2))
            aff_homo_ss_sum += 1.0 - norm / thred_homo
            aff_homo_ss_cnt += 1

    return [aff_ss_sum / aff_ss_cnt, aff_homo_ss_sum / aff_homo_ss_cnt]

def bbox_overlap_rate(bboxes_s, bboxes_t):
    """
    Compute overlap rate between bounding boxes.
    
    Args:
        bboxes_s (np.ndarray): Source bounding boxes, shape (num_s, 4)
        bboxes_t (np.ndarray): Target bounding boxes, shape (num_t, 4)
    
    Returns:
        np.ndarray: Overlap rates, shape (num_s, num_t)
    """
    num_s, num_t = bboxes_s.shape[0], bboxes_t.shape[0]
    validate_array_shape(bboxes_s, [-1, 4], "bboxes_s")
    validate_array_shape(bboxes_t, [-1, 4], "bboxes_t")

    overlap_rate = np.zeros((num_s, num_t))
    for i in range(num_s):
        area_s = (bboxes_s[i, 2] - bboxes_s[i, 0]) * (bboxes_s[i, 3] - bboxes_s[i, 1])
        for j in range(num_t):
            x_left = np.maximum(bboxes_s[i, 0], bboxes_t[j, 0])
            y_top = np.maximum(bboxes_s[i, 1], bboxes_t[j, 1])
            x_right = np.minimum(bboxes_s[i, 2], bboxes_t[j, 2])
            y_bottom = np.minimum(bboxes_s[i, 3], bboxes_t[j, 3])

            if x_right < x_left or y_bottom < y_top:
                overlap_rate[i, j] = 0.0
            else:
                intersect = (x_right - x_left) * (y_bottom - y_top)
                overlap_rate[i, j] = intersect / (area_s + EPSILON)
    
    return overlap_rate