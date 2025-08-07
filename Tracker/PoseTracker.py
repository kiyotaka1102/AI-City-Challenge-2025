from functools import partial
import numpy as np
import cv2
import sys

from Tracker.kalman_filter_box_zya import KalmanFilter_box, KalmanFilter_position3D
from Solver.bip_solver import GLPKSolver
from util.camera import *
from util.process import find_view_for_cluster
from scipy.optimize import linear_sum_assignment
from Tracker.matching import *
import lap
import aic_cpp
from util.func_3d_bbox import *
from util.center import *
from collections import deque
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cosine, pdist, squareform
from numpy.linalg import norm
from scipy.optimize import minimize, differential_evolution, approx_fprime
from scipy.stats import multivariate_normal
from collections import Counter


class TrackState:
    """
    使用TrackState表示追踪轨迹的状态
    """
    Unconfirmed = 1
    Confirmed = 2
    Missing = 3
    Deleted = 4

class Track2DState:
    Vide = 0
    Detected = 1
    Occluded = 2
    Missing = 3


class Detection_Sample:
    def __init__(self, bbox, keypoints_2d, reid_feat, cam_id, frame_id, class_id=None, obj_id=None, trajectory_center=None, yaw=None, width=None, length=None, height=None, yaw3d=None, bbox_center3d=None, depth_frame=None):
        self.bbox = bbox
        self.keypoints_2d = keypoints_2d
        self.reid_feat = reid_feat
        self.cam_id = cam_id
        self.frame_id = frame_id
        self.class_id = int(class_id) if class_id is not None else 0
        self.obj_id = obj_id
        self.trajectory_center = trajectory_center
        self.yaw2d = yaw
        self.width = width
        self.length = length
        self.height = height
        self.yaw3d = yaw3d
        self.bbox_center3d = bbox_center3d
        # self.depth_frame= depth_frame

class PoseTrack2D:
    def __init__(self):
        self.state = []
        self.bbox = None
        self.keypoints_2d = None
        self.reid_feat = None
        self.cam_id = None
        self.class_id = -1
        self.width = None
        self.length = None
        self.height = None
        self.yaw3d = None

    def init_with_det(self, Detect_Sample):
        if len(self.state) == 10:
            self.state.pop()
        self.state = [Track2DState.Detected] + self.state
        self.bbox = Detect_Sample.bbox
        self.keypoints_2d = Detect_Sample.keypoints_2d
        self.reid_feat = Detect_Sample.reid_feat
        self.cam_id = Detect_Sample.cam_id
        self.class_id = int(Detect_Sample.class_id)
        self.width = Detect_Sample.width
        self.length = Detect_Sample.length
        self.height = Detect_Sample.height
        self.yaw3d = Detect_Sample.yaw3d


class PoseTrack:

    def __init__(self, cameras):
        self.id = -1
        self.class_id = -1
        self.cameras = cameras
        self.num_cam = len(cameras)
        self.confirm_time_left = 0
        self.valid_views = []
        self.bank_size = 100
        self.feat_bank = np.zeros((self.bank_size, 2048), dtype=np.float16)  # Use float16 for feature bank
        self.feat_count = 0
        self.track2ds = [PoseTrack2D() for _ in range(self.num_cam)]
        self.state = TrackState.Unconfirmed
        self.num_keypoints = 17
        self.keypoint_thrd = np.float16(0.6)
        self.update_age = 0
        self.decay_weight = np.float16(0.5)
        self.keypoints_3d = np.zeros((self.num_keypoints, 4), dtype=np.float16)  # Use float16 for 3D keypoints
        self.keypoints_mv = np.zeros((self.num_cam, self.num_keypoints, 3), dtype=np.float16)  # Use float16 for multi-view keypoints
        self.bbox_mv = np.zeros((self.num_cam, 5), dtype=np.float16)  # Use float16 for multi-view bboxes
        self.age_2D = np.ones((self.num_cam, self.num_keypoints), dtype=np.float16) * np.inf
        self.age_3D = np.ones(self.num_keypoints, dtype=np.float16) * np.inf
        self.age_bbox = np.ones(self.num_cam, dtype=np.float16) * np.inf
        self.dura_bbox = np.zeros(self.num_cam, dtype=np.int16)  # Use int16 for duration
        self.thred_conf_reid = np.float16(0.95)
        self.feet_idx = np.array([15, 16], dtype=np.int16)
        self.bbox_kalman = [KalmanFilter_box() for _ in range(self.num_cam)]
        self.thred_reid = np.float16(0.5)
        self.thred_reid_robot = np.float16(0.7)
        self.history = deque(maxlen=10)  # History size of 10
        self.frechet_threshold = np.float16(1.0)
        self.output_cord = np.zeros(3, dtype=np.float16)
        self.output_priority = [[5, 6], [11, 12], [15, 16]]
        self.main_joints = np.array([5, 6, 11, 12, 13, 14, 15, 16], dtype=np.int16)
        self.joint_pairs = [(5, 6), (11, 12), (15, 16)]
        self.upper_body = np.array([5, 6, 11, 12], dtype=np.int16)
        self.sample_buf = []
        self.unit = np.full((self.num_keypoints, 3), np.float16(1 / np.sqrt(3)), dtype=np.float16)
        self.iou_mv = [np.float16(0) for _ in range(self.num_cam)]
        self.ovr_mv = [np.float16(0) for _ in range(self.num_cam)]
        self.oc_state = [False for _ in range(self.num_cam)]
        self.oc_idx = [[] for _ in range(self.num_cam)]
        self.ovr_tgt_mv = [np.float16(0) for _ in range(self.num_cam)]
        self.width_mv = np.zeros(self.num_cam, dtype=np.float16)
        self.height_mv = np.zeros(self.num_cam, dtype=np.float16)
        self.length_mv = np.zeros(self.num_cam, dtype=np.float16)
        self.yaw3d_mv = np.zeros(self.num_cam, dtype=np.float16)
        self.bbox_center3d_mv = np.zeros((self.num_cam, 3), dtype=np.float16)
        self.width = None
        self.height = None
        self.length = None
        self.yaw3d = None
        self.bbox_center3d = np.zeros(3, dtype=np.float16)
        self.center_kalman = KalmanFilter_position3D(dt=np.float32(1.0 / 30.0))  # Use float32 for Kalman filter
    
    def update_trajectory(self):
        if self.output_cord[-1] > 0 and not np.any(np.isnan(self.output_cord[:2])) and not np.any(np.isinf(self.output_cord[:2])):
            self.history.append(self.output_cord[:2].copy().astype(np.float16))
            if len(self.history) > 10:
                self.history.popleft()
        
    def _is_valid_joint_pairs(self, v):
        """Check if at least one joint pair has at least 2 keypoint with confidence score > 0.3 for view v."""
        for left_idx, right_idx in self.joint_pairs:
            if self.keypoints_mv[v][left_idx, 2] > 0 and self.keypoints_mv[v][right_idx, 2] > 0:
                return True
        return False

    def _is_valid_sample_joint_pairs(self, keypoints_2d):
        """Check if at least one joint pair has at least 2 keypoint with confidence score > 0.3 for a sample."""
        for left_idx, right_idx in self.joint_pairs:
            if keypoints_2d[left_idx, -1] > 0 and keypoints_2d[right_idx, -1] > 0:
                return True
        return False
   
    
    def robust_initial_guess(self, filtered_views):
        pairwise_centers = []
        for i, v1 in enumerate(filtered_views):
            for v2 in filtered_views[i+1:]:
                cam1, cam2 = self.cameras[v1], self.cameras[v2]
                bbox1, bbox2 = self.bbox_mv[v1], self.bbox_mv[v2]
                center_2d_1 = np.array([(bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2])
                center_2d_2 = np.array([(bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2])
                point1 = np.linalg.pinv(cam1.project_mat) @ np.array([center_2d_1[0], center_2d_1[1], 1])
                point2 = np.linalg.pinv(cam2.project_mat) @ np.array([center_2d_2[0], center_2d_2[1], 1])
                ray1 = (point1[:3]/point1[-1] - cam1.pos) / np.linalg.norm(point1[:3]/point1[-1] - cam1.pos)
                ray2 = (point2[:3]/point2[-1] - cam2.pos) / np.linalg.norm(point2[:3]/point2[-1] - cam2.pos)
                def error(p):
                    return (np.linalg.norm(np.cross(p - cam1.pos, ray1))**2 + 
                            np.linalg.norm(np.cross(p - cam2.pos, ray2))**2)
                result = minimize(error, x0=(cam1.pos + cam2.pos)/2)
                if result.success:
                    pairwise_centers.append(result.x)
        if pairwise_centers:
            return np.median(pairwise_centers, axis=0)
        return self.bbox_center3d.copy()
    

    def refine_3d_center(self, valid_views, max_iter=100, tol=1e-6, bounds_margin=10.0, 
                        area_ratio_threshold=0.15, conf_threshold=0.6, error_threshold=1.0, 
                        reproj_error_threshold=2.0):
        """
        Refine the 3D bounding box center by minimizing distance to rays from 2D centers.
        Excludes views where the 2D bounding box area is too small or confidence is too low.
        Applies temporal smoothing with adaptive Kalman filter updates.
        Adds reprojection error constraint to handle occlusions and improve 2D-3D alignment.

        Args:
            valid_views (list): List of camera view indices with valid bounding boxes (age_bbox == 0).
            max_iter (int): Maximum iterations for optimization.
            tol (float): Tolerance for optimization convergence.
            bounds_margin (float): Margin for spatial bounds around initial center (meters).
            area_ratio_threshold (float): Minimum area ratio relative to the largest bbox.
            conf_threshold (float): Minimum confidence score for a bounding box.
            error_threshold (float): Distance threshold (meters) for detecting large position errors.
            reproj_error_threshold (float): Pixel threshold for detecting occluded views based on reprojection error.

        Returns:
            np.ndarray: Smoothed 3D center coordinates (shape: (3,)).
        """
        if len(valid_views) < 1 or np.all(self.bbox_center3d == 0):
            self.center_kalman.predict()
            return self.center_kalman.get_state()

        # Calculate bounding box areas and collect confidence scores
        areas = []
        for v in valid_views:
            bbox = self.bbox_mv[v]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            conf = bbox[-1]
            areas.append((v, area, conf))
        
        # Filter views based on area ratio and confidence
        max_area = max(area for _, area, _ in areas) if areas else 1.0
        filtered_views = [
            v for v, area, conf in areas
            if  area >= area_ratio_threshold * max_area and conf >= conf_threshold
        ]
        # filtered_views = valid_views
        if len(filtered_views) < 1:
            self.center_kalman.predict()
            return self.center_kalman.get_state()

        # Initialize with weighted triangulation based on confidence and area
        initial_center = self.bbox_center3d.copy()
        if np.all(initial_center == 0):
            rays = []
            origins = []
            centers_2d = []
            weights = []
            for v in filtered_views:
                bbox = self.bbox_mv[v]
                center_2d = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                centers_2d.append(center_2d)
                cam = self.cameras[v]
                point_homo = np.linalg.pinv(cam.project_mat) @ np.array([center_2d[0], center_2d[1], 1])
                if abs(point_homo[-1]) > 1e-5:
                    point_3d = point_homo[:3] / point_homo[-1]
                    ray = point_3d - cam.pos
                    ray /= np.linalg.norm(ray) + 1e-6
                    rays.append(ray)
                    origins.append(cam.pos)
                    weights.append(bbox[-1] * np.sqrt(self.bbox_mv[v][2] - self.bbox_mv[v][0]))
                else:
                    weights.append(0.0)
            
            if len(rays) >= 2:
                def weighted_ray_error(point_3d):
                    error = 0.0
                    for origin, ray, w in zip(origins, rays, weights):
                        dist = np.linalg.norm(np.cross(point_3d - origin, ray)) / (np.linalg.norm(ray) + 1e-6)
                        error += w * dist ** 2
                    return error / (sum(weights) + 1e-6)
                result = minimize(
                    weighted_ray_error,
                    x0=np.mean(origins, axis=0),
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'fatol': 1e-6}
                )
                if result.success:
                    initial_center = result.x
                else:
                    points_2d = []
                    for v, center_2d in zip(filtered_views, centers_2d):
                        cam = self.cameras[v]
                        point_homo = cam.homo_inv @ np.array([center_2d[0], center_2d[1], 1])
                        if abs(point_homo[-1]) > 1e-5:
                            point_2d = point_homo[:-1] / point_homo[-1]
                            points_2d.append(point_2d)
                    if points_2d:
                        initial_center = np.append(np.mean(points_2d, axis=0), 0.0)
                    else:
                        self.center_kalman.predict()
                        return self.center_kalman.get_state()
        
        def epipolar_optimization_error(center_3d, views, bboxes_2d, cameras):
            # print(f"center_3d: {center_3d.tolist()}, views: {views}")
            weights = []
            rays = []
            origins = []
            centers_2d = []

            # Precompute rays and weights
            for v, bbox in zip(views, bboxes_2d):
                cam = cameras[v]
                center_2d = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                centers_2d.append(center_2d)
                point_homo = np.linalg.pinv(cam.project_mat) @ np.array([center_2d[0], center_2d[1], 1])
                if abs(point_homo[-1]) < 1e-8:
                    weights.append(0.1)
                    rays.append(np.zeros(3))
                    origins.append(cam.pos)
                    print(f"View {v}: Invalid point_homo, using zero ray")
                    continue
                point_3d = point_homo[:3] / point_homo[-1]
                ray = point_3d - cam.pos
                ray /= np.linalg.norm(ray) + 1e-6
                rays.append(ray)
                origins.append(cam.pos)
                conf = max(bbox[-1], 0.1)
                # ray_length = np.linalg.norm(point_3d - cam.pos) + 1e-6
                weights.append(conf)
                # print(f"View {v}: center_2d={center_2d.tolist()}, ray={ray.tolist()}, origin={cam.pos.tolist()}, weight={weights[-1]:.4f}")

            # Compute epipolar error
            epi_error = 0.0
            total_weight = 0.0
            # if len(views) < 2:
            #     # print("Only one view, skipping epipolar score calculation")
            if len(views) >2:
                for i in range(len(views)):
                    for j in range(i + 1, len(views)):
                        if weights[i] < 0.2 or weights[j] < 0.2:
                            # print(f"Pair ({views[i]}, {views[j]}): Skipped due to low weights ({weights[i]:.4f}, {weights[j]:.4f})")
                            continue
                        # print(f"Pair ({views[i]}, {views[j]}): ray_shapes={rays[i].shape}, {rays[j].shape}")
                        score = epipolar_3d_score_norm(origins[i], rays[i].reshape(1, 3), origins[j], rays[j].reshape(1, 3), 1)
                        # print(f"View {v}: center_2d={center_2d.tolist()}, ray={ray.tolist()}, origin={cam.pos.tolist()}, weight={weights[-1]:.4f}")
                        epi_error += weights[i] * weights[j] * (1.0 - score) ** 2
                        total_weight += weights[i] * weights[j]

            if total_weight > 0:
                epi_error /= total_weight + 1e-6

            # Compute reprojection error
            reproj_error = 0.0
            for k, (bbox, center_2d) in enumerate(zip(bboxes_2d, centers_2d)):
                cam = cameras[views[k]]
                proj_homo = cam.project_mat @ np.append(center_3d, 1)
                if abs(proj_homo[-1]) > 1e-5:
                    proj_2d = proj_homo[:2] / proj_homo[-1]
                    reproj_err = np.linalg.norm(proj_2d - center_2d)
                    reproj_weight = 1.0 / (1.0 + 0.1 * reproj_err)
                else:
                    reproj_err = 1e6
                    reproj_weight = 0.1
                # print(reproj_weight)
                reproj_error += weights[k] * reproj_weight * reproj_err ** 2
            if sum(weights) > 0:
                reproj_error /= sum(weights) + 1e-6

            # Combine errors
            deviation = np.linalg.norm(center_3d - initial_center)
            total_error = 0.4 * epi_error +  0.2* reproj_error + 0.1 * deviation ** 2
            # print(f"Total error: {total_error}, epi_error={epi_error}, reproj_error={reproj_error}, deviation={deviation}")
            return 1e6 if not np.isfinite(total_error) else total_error    
        # Run optimization
        bboxes_2d = [self.bbox_mv[v] for v in filtered_views]
        bounds = [
            (initial_center[0] - bounds_margin, initial_center[0] + bounds_margin),
            (initial_center[1] - bounds_margin, initial_center[1] + bounds_margin),
            (initial_center[2] - 2, initial_center[2] + 2)
        ]
        result = minimize(
            fun=epipolar_optimization_error,
            x0=initial_center,
            args=(filtered_views, bboxes_2d, self.cameras),
            method='Nelder-Mead', 
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol}
        )
        # Predict Kalman filter state
        self.center_kalman.predict()

        if result.success:
            refined_center = result.x
            if (np.abs(refined_center) > 1e6).any() or np.linalg.norm(refined_center - initial_center) >  bounds_margin:
                return self.center_kalman.get_state() 
            
            # Check reprojection errors to detect occlusion
            trusted_reproj_errors = []
            trusted_views = []
            for v, bbox in zip(filtered_views, bboxes_2d):
                cam = self.cameras[v]
                center_2d = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                proj_homo = cam.project_mat @ np.append(refined_center, 1)
                if abs(proj_homo[-1]) > 1e-5:
                    proj_2d = proj_homo[:2] / proj_homo[-1]
                    reproj_error = np.linalg.norm(proj_2d - center_2d)
                    # print(reproj_error)
                    if reproj_error < reproj_error_threshold:  # Only include reliable views
                        trusted_reproj_errors.append(reproj_error)
                        trusted_views.append(v)
            
            # Check for large position error or excessive reprojection errors
            predicted_center = self.center_kalman.get_state()
            error = np.linalg.norm(refined_center - predicted_center)
            avg_conf = np.mean([self.bbox_mv[v][-1] for v in trusted_views])

            if error > error_threshold or len(trusted_views) == 0:
                if len(trusted_views) <= 1:
                    self.center_kalman.reset()
                    self.center_kalman.init_state(refined_center, high_covariance=True)
                    return refined_center
                else:
                    R_scale = 10.0 
            else:
                R_scale = 1.0 / (len(trusted_views) * max(avg_conf, 0.1))

            # Initialize Kalman filter if state is zero
            if np.all(self.center_kalman.get_state() == 0):
                self.center_kalman.init_state(refined_center, high_covariance=True)

            # Update Kalman filter with adaptive noise
            R = np.diag([R_scale, R_scale, R_scale * 2])
            self.center_kalman.update(refined_center, R=R, num_views=len(trusted_views), avg_conf=avg_conf)

            return self.center_kalman.get_state()
        else:
            return self.center_kalman.get_state()


    # def refine_3d_center(self, valid_views, max_iter=50, tol=1e-6, bounds_margin=10.0,
    #                  area_ratio_threshold=0.15, conf_threshold=0.8, error_threshold=5.0,
    #                  reproj_error_threshold=50.0, heatmap_sigma_scale=0.25):
    #     """
    #     Refine the 3D bounding box center using heatmap-based optimization.
    #     Args:
    #         ... (same as your original args)
    #         heatmap_sigma_scale (float): Scaling factor for Gaussian heatmap standard deviation.
    #     Returns:
    #         np.ndarray: Smoothed 3D center coordinates (shape: (3,)).
    #     """
    #     if len(valid_views) < 1 or np.all(self.bbox_center3d == 0):
    #         self.center_kalman.predict()
    #         return self.center_kalman.get_state()

    #     # Calculate bounding box areas and collect confidence scores
    #     areas = []
    #     for v in valid_views:
    #         bbox = self.bbox_mv[v]
    #         width = bbox[2] - bbox[0]
    #         height = bbox[3] - bbox[1]
    #         area = width * height
    #         conf = bbox[-1]
    #         areas.append((v, area, conf))
        
    #     # Filter views based on area ratio and confidence
    #     if areas:
    #         max_area = max(area for _, area, _ in areas)
    #         filtered_views = [
    #             v for v, area, conf in areas
    #             if area >= area_ratio_threshold * max_area and conf >= conf_threshold
    #         ]
    #     else:
    #         filtered_views = valid_views

    #     if len(filtered_views) < 1:
    #         self.center_kalman.predict()
    #         return self.center_kalman.get_state()

    #     # Initialize heatmaps for each view
    #     heatmaps = []
    #     centers_2d = []
    #     for v in filtered_views:
    #         bbox = self.bbox_mv[v]
    #         center_2d = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    #         centers_2d.append(center_2d)
    #         # Compute Gaussian heatmap
    #         width = bbox[2] - bbox[0]
    #         height = bbox[3] - bbox[1]
    #         sigma_x = width * heatmap_sigma_scale
    #         sigma_y = height * heatmap_sigma_scale
    #         cov = np.diag([sigma_x**2, sigma_y**2])
    #         # Assume image dimensions (adjust as needed)
    #         x, y = np.meshgrid(np.arange(0, 1920), np.arange(0, 1080))  # Example resolution
    #         pos = np.dstack((x, y))
    #         gaussian = multivariate_normal(mean=center_2d, cov=cov)
    #         heatmap = gaussian.pdf(pos)
    #         heatmap /= heatmap.max() + 1e-6  # Normalize
    #         heatmaps.append(heatmap)

    #     # Initialize with current 3D center or triangulate
    #     initial_center = self.bbox_center3d.copy()
    #     if np.all(initial_center == 0):
    #         # Fallback: Triangulate as in your original code
    #         rays = []
    #         origins = []
    #         for v, center_2d in zip(filtered_views, centers_2d):
    #             cam = self.cameras[v]
    #             point_homo = np.linalg.pinv(cam.project_mat) @ np.array([center_2d[0], center_2d[1], 1])
    #             if abs(point_homo[-1]) > 1e-5:
    #                 point_3d = point_homo[:3] / point_homo[-1]
    #                 ray = point_3d - cam.pos
    #                 ray /= np.linalg.norm(ray) + 1e-6
    #                 rays.append(ray)
    #                 origins.append(cam.pos)
    #         if len(rays) >= 2:
    #             def ray_error(point_3d):
    #                 error = 0.0
    #                 for origin, ray in zip(origins, rays):
    #                     dist = np.linalg.norm(np.cross(point_3d - origin, ray)) / (np.linalg.norm(ray) + 1e-6)
    #                     error += dist ** 2
    #                 return error
    #             result = minimize(
    #                 ray_error,
    #                 x0=np.mean(origins, axis=0),
    #                 method='Powell',
    #                 options={'maxiter': 100, 'ftol': 1e-6}
    #             )
    #             if result.success:
    #                 initial_center = result.x
    #             else:
    #                 self.center_kalman.predict()
    #                 return self.center_kalman.get_state()

    #     def heatmap_loss(center_3d, views, bboxes_2d, cameras, heatmaps):
    #         score = 0.0
    #         reproj_errors = []
    #         for v, bbox, heatmap in zip(views, bboxes_2d, heatmaps):
    #             cam = cameras[v]
    #             center_2d = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    #             proj_homo = cam.project_mat @ np.append(center_3d, 1)
    #             if abs(proj_homo[-1]) < 1e-5:
    #                 score -= 1e6
    #                 continue
    #             proj_2d = proj_homo[:2] / proj_homo[-1]
    #             # Ensure proj_2d is within image bounds
    #             proj_2d = np.clip(proj_2d, [0, 0], [1919, 1079])  # Example resolution
    #             heatmap_value = heatmap[int(proj_2d[1]), int(proj_2d[0])]
    #             reproj_error = np.linalg.norm(proj_2d - center_2d)
    #             reproj_errors.append(reproj_error)
    #             reproj_weight = 1.0 if reproj_error < reproj_error_threshold else reproj_error_threshold / (reproj_error + 1e-6)
    #             weight = max(bbox[-1], 0.1) * reproj_weight
    #             score += weight * heatmap_value
    #         # Add deviation penalty
    #         deviation = np.linalg.norm(center_3d - initial_center)
    #         score -= 0.1 * deviation ** 2
    #         return -score  # Minimize negative score

    #     # Run optimization
    #     bboxes_2d = [self.bbox_mv[v] for v in filtered_views]
    #     bounds = [
    #         (initial_center[0] - bounds_margin, initial_center[0] + bounds_margin),
    #         (initial_center[1] - bounds_margin, initial_center[1] + bounds_margin),
    #         (initial_center[2] - 5.0, initial_center[2] + 5.0)
    #     ]
    #     result = minimize(
    #         fun=heatmap_loss,
    #         x0=initial_center,
    #         args=(filtered_views, bboxes_2d, self.cameras, heatmaps),
    #         method='Powell',
    #         bounds=bounds,
    #         options={'maxiter': max_iter, 'ftol': tol}
    #     )

    #     # Predict Kalman filter state
    #     self.center_kalman.predict()

    #     if result.success:
    #         refined_center = result.x
    #         if np.any(np.isnan(refined_center)) or np.linalg.norm(refined_center - initial_center) > 1.5 * bounds_margin:
    #             return self.center_kalman.get_state()

    #         # Compute average heatmap score for R adjustment
    #         avg_score = 0.0
    #         reproj_errors = []
    #         for v, bbox, heatmap in zip(filtered_views, bboxes_2d, heatmaps):
    #             cam = self.cameras[v]
    #             center_2d = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    #             proj_homo = cam.project_mat @ np.append(refined_center, 1)
    #             if abs(proj_homo[-1]) > 1e-5:
    #                 proj_2d = proj_homo[:2] / proj_homo[-1]
    #                 proj_2d = np.clip(proj_2d, [0, 0], [1919, 1079])
    #                 avg_score += heatmap[int(proj_2d[1]), int(proj_2d[0])]
    #                 reproj_error = np.linalg.norm(proj_2d - center_2d)
    #                 reproj_errors.append(reproj_error)
    #         avg_score /= max(1, len(filtered_views))

    #         predicted_center = self.center_kalman.get_state()
    #         error = np.linalg.norm(refined_center - predicted_center)
    #         avg_reproj_error = np.mean(reproj_errors) if reproj_errors else 0.0
    #         if error > error_threshold or avg_reproj_error > reproj_error_threshold:
    #             if len(filtered_views) <= 1 or avg_reproj_error > reproj_error_threshold * 2:
    #                 self.center_kalman.reset()
    #                 self.center_kalman.init_state(refined_center, high_covariance=True)
    #                 return refined_center
    #             else:
    #                 R_scale = 10.0
    #         else:
    #             avg_conf = np.mean([self.bbox_mv[v][-1] for v in filtered_views])
    #             R_scale = 1.0 / (len(filtered_views) * max(avg_score, 0.1) * max(avg_conf, 0.1))

    #         if np.all(self.center_kalman.get_state() == 0):
    #             self.center_kalman.init_state(refined_center, high_covariance=True)

    #         R = np.diag([R_scale, R_scale, R_scale * 2])
    #         self.center_kalman.update(refined_center, R)
    #         return self.center_kalman.get_state()
    #     else:
    #         return self.center_kalman.get_state()
    
    
    def optimize_3d_bbox(self, valid_views, max_iter=100, tol=1e-6, alpha_epi=0.2):
        """
        Optimize 3D bounding box dimensions (width, height, length) by minimizing reprojection errors across multiple views.
        Uses epipolar geometry for view weighting and robust multi-view fusion. Keeps 3D center and yaw fixed.
        
        Args:
            valid_views (list): List of camera view indices with valid bounding boxes.
            max_iter (int): Maximum iterations for optimization.
            tol (float): Tolerance for convergence.
            alpha_epi (float): Scaling factor for epipolar scoring.
        
        Returns:
            tuple: (center, width, height, length, yaw) of the optimized 3D bounding box, with center and yaw unchanged.
        """
        if len(valid_views) < 2:
            return self.bbox_center3d, self.width, self.height, self.length, self.yaw3d

        # Initialize parameters
        initial_width = self.width if self.width is not None else 1.0
        initial_height = self.height if self.height is not None else 1.0
        initial_length = self.length if self.length is not None else 1.0
        fixed_center = self.bbox_center3d.copy() if np.any(self.bbox_center3d) else np.zeros(3)
        fixed_yaw = self.yaw3d if self.yaw3d is not None else 0.0

        initial_params = np.array([
            initial_width, initial_height, initial_length
        ])

        def reprojection_error(params, views, bboxes_2d, cameras, T, yaw):
            width, height, length = params
            error = 0.0

            # Define rotation matrix using fixed yaw
            R = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])

            # Define 3D box corners
            dx, dy, dz = length, width, height
            corners = np.array([
                [ dx/2,  dy/2,  dz/2], [-dx/2,  dy/2,  dz/2],
                [-dx/2, -dy/2,  dz/2], [ dx/2, -dy/2,  dz/2],
                [ dx/2,  dy/2, -dz/2], [-dx/2,  dy/2, -dz/2],
                [-dx/2, -dy/2, -dz/2], [ dx/2, -dy/2, -dz/2]
            ])

            # Compute pairwise epipolar scores for view weighting
            epipolar_weights = []
            for i, v1 in enumerate(views):
                cam1 = cameras[v1]
                p1 = cam1.pos
                # Estimate ray from 2D bounding box center
                bbox_center = np.array([
                    (bboxes_2d[i][0] + bboxes_2d[i][2]) / 2,
                    (bboxes_2d[i][1] + bboxes_2d[i][3]) / 2,
                    1.0
                ])
                ray1 = (cam1.project_inv @ bbox_center)[:3]
                ray1 /= np.linalg.norm(ray1) + 1e-6

                score_sum = 0.0
                for j, v2 in enumerate(views):
                    if j <= i:
                        continue
                    cam2 = cameras[v2]
                    p2 = cam2.pos
                    bbox_center2 = np.array([
                        (bboxes_2d[j][0] + bboxes_2d[j][2]) / 2,
                        (bboxes_2d[j][1] + bboxes_2d[j][3]) / 2,
                        1.0
                    ])
                    ray2 = (cam2.project_inv @ bbox_center2)[:3]
                    ray2 /= np.linalg.norm(ray2) + 1e-6
                    score_sum += epipolar_3d_score_norm(p1, ray1.reshape(1, 3), p2, ray2.reshape(1, 3), alpha_epi)
                epipolar_weights.append(score_sum / max(1, len(views) - 1))

            for v, bbox_2d, epi_weight in zip(views, bboxes_2d, epipolar_weights):
                cam = cameras[v]
                xmin, ymin, xmax, ymax, score = bbox_2d
                # Combine detection confidence and epipolar score
                weight = max(score, 0.1) * max(epi_weight, 0.1)

                # Use project_mat for projection
                proj_mat = cam.project_mat

                # Project corners to 2D
                projected = []
                for corner in corners:
                    X_world = R @ corner + T
                    X_homo = proj_mat @ np.append(X_world, 1)
                    if abs(X_homo[-1]) < 1e-5:
                        continue  # Skip invalid projections
                    x = X_homo[:2] / X_homo[-1]
                    # Check if projection is within frame bounds
                    frame_width = 1920
                    frame_height = 1080
                    if 0 <= x[0] <= frame_width and 0 <= x[1] <= frame_height:
                        projected.append(x)
                
                if not projected:
                    continue  # Skip view if no valid projections
                projected = np.array(projected)

                # Compute reprojection errors
                proj_xmin, proj_ymin = np.min(projected, axis=0)
                proj_xmax, proj_ymax = np.max(projected, axis=0)
                errors = [
                    (proj_xmin - xmin) ** 2,
                    (proj_xmax - xmax) ** 2,
                    (proj_ymin - ymin) ** 2,
                    (proj_ymax - ymax) ** 2
                ]
                view_error = weight * sum(errors)
                error += view_error

            # Regularization: Penalize large deviations from initial values
            error += 0.1 * (
                (width - initial_width) ** 2 +
                (height - initial_height) ** 2 +
                (length - initial_length) ** 2
            )

            return error

        # Define bounds for dimensions only
        bounds = [
            (0.1, max(initial_width * 3.0, 1.0)),
            (0.1, max(initial_height * 3.0, 1.0)),
            (0.1, max(initial_length * 3.0, 1.0))
        ]

        # Optimize
        bboxes_2d = [self.bbox_mv[v] for v in valid_views]
        result = minimize(
            reprojection_error,
            x0=initial_params,
            args=(valid_views, bboxes_2d, self.cameras, fixed_center, fixed_yaw),
            method='Powell',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol}
        )

        if result.success and not np.any(np.isnan(result.x)):
            w, h, l = result.x
            if w > 0.1 and h > 0.1 and l > 0.1:  # Ensure valid dimensions
                return self.bbox_center3d, w, h, l, self.yaw3d

        return self.bbox_center3d, self.width, self.height, self.length, self.yaw3d
        
    def reactivate(self, newtrack):
        self.state = newtrack.state
        self.valid_views = newtrack.valid_views
        self.track2ds = newtrack.track2ds
        self.age_2D = newtrack.age_2D
        self.age_3D = newtrack.age_3D
        self.bbox_mv = newtrack.bbox_mv
        self.age_bbox = newtrack.age_bbox
        self.confirm_time_left = newtrack.confirm_time_left
        self.keypoints_3d = newtrack.keypoints_3d
        self.keypoints_mv = newtrack.keypoints_mv
        self.output_cord = newtrack.output_cord
        self.dura_bbox = newtrack.dura_bbox
        self.bbox_kalman = newtrack.bbox_kalman
        self.update_age = 0
        for v in range(self.num_cam):   
            if self._is_valid_joint_pairs(v):
                self.width_mv[v] = newtrack.width_mv[v]
                self.height_mv[v] = newtrack.height_mv[v]
                self.length_mv[v] = newtrack.length_mv[v]
                self.yaw3d_mv[v] = newtrack.yaw3d_mv[v]
                self.bbox_center3d_mv[v] = newtrack.bbox_center3d_mv[v]
        # Copy fused 3D parameters only if at least one view has sufficient keypoint scores
        valid_views = [v for v in newtrack.valid_views if newtrack._is_valid_joint_pairs(v)]
        if valid_views:
            self.width = newtrack.width
            self.height = newtrack.height
            self.length = newtrack.length
            self.yaw3d = newtrack.yaw3d
            self.bbox_center3d = newtrack.bbox_center3d

        if newtrack.feat_count >= 1:
            if self.feat_count >= self.bank_size:
                bank = self.feat_bank
            else:
                bank = self.feat_bank[:self.feat_count % self.bank_size]

            new_bank = newtrack.feat_bank[:newtrack.feat_count % self.bank_size]
            sim = np.max((new_bank @ bank.T), axis=-1)
            sim_idx = np.where(sim < self.thred_reid)[0]
            for id in sim_idx:
                self.feat_bank[self.feat_count % self.bank_size] = new_bank[id].copy()
                self.feat_count += 1

    def switch_view(self, track, v):
        self.track2ds[v], track.track2ds[v] = track.track2ds[v], self.track2ds[v]
        self.age_2D[v], track.age_2D[v] = track.age_2D[v], self.age_2D[v]
        self.keypoints_mv[v], track.keypoints_mv[v] = track.keypoints_mv[v], self.keypoints_mv[v]
        self.age_bbox[v], track.age_bbox[v] = track.age_bbox[v], self.age_bbox[v]
        self.bbox_mv[v], track.bbox_mv[v] = track.bbox_mv[v], self.bbox_mv[v]
        self.dura_bbox[v], track.dura_bbox[v] = track.dura_bbox[v], self.dura_bbox[v]
        self.oc_state[v], track.oc_state[v] = track.oc_state[v], self.oc_state[v]
        self.oc_idx[v], track.oc_idx[v] = track.oc_idx[v], self.oc_idx[v]
        self.bbox_kalman[v], track.bbox_kalman[v] = track.bbox_kalman[v], self.bbox_kalman[v]
        self.iou_mv[v], track.iou_mv[v] = track.iou_mv[v], self.iou_mv[v]
        self.ovr_mv[v], track.ovr_mv[v] = track.ovr_mv[v], self.ovr_mv[v]
        self.ovr_tgt_mv[v], track.ovr_tgt_mv[v] = track.ovr_tgt_mv[v], self.ovr_tgt_mv[v]
        self.width_mv[v], track.width_mv[v] = track.width_mv[v], self.width_mv[v]
        self.height_mv[v], track.height_mv[v] = track.height_mv[v], self.height_mv[v]
        self.length_mv[v], track.length_mv[v] = track.length_mv[v], self.length_mv[v]
        self.yaw3d_mv[v], track.yaw3d_mv[v] = track.yaw3d_mv[v], self.yaw3d_mv[v]
        self.bbox_center3d_mv[v], track.bbox_center3d_mv[v] = track.bbox_center3d_mv[v], self.bbox_center3d_mv[v]


    def get_output(self):
        # Compute fused 3D dimensions and yaw by averaging valid views with sufficient keypoint scores
        
        valid_views = [v for v in self.valid_views if self.age_bbox[v] == 0]
        # Assign class_id based on valid views
        if valid_views:
            class_ids = [self.track2ds[v].class_id for v in valid_views if self.track2ds[v].class_id is not None]
            if class_ids:            
                if 5 in class_ids:
                    self.class_id = 5
                elif 4 in class_ids:
                    self.class_id = 4
                else:
                    class_count = Counter(class_ids)
                    self.class_id = class_count.most_common(1)[0][0]
        if valid_views:
            # Collect features
            valid_widths = [(v, self.width_mv[v]) for v in valid_views if self.width_mv[v] is not None and self.bbox_mv[v][-1] >0.7]
            valid_heights = [(v, self.height_mv[v]) for v in valid_views if self.height_mv[v] is not None and self.bbox_mv[v][-1] >0.7]
            valid_lengths = [(v, self.length_mv[v]) for v in valid_views if self.length_mv[v] is not None and self.bbox_mv[v][-1] >0.7]
            valid_yaw3ds = [(v, self.yaw3d_mv[v]) for v in valid_views if self.yaw3d_mv[v] is not None]
            valid_bbox_centers = [
                (v, self.bbox_center3d_mv[v]) for v in valid_views
                # if self.bbox_center3d_mv[v] is not None and self.bbox_mv[v][-1] > 0.7
            ]
            
            # Function to select view with lowest avg_dists
            def select_lowest_avg_dist(values, name, is_3d=False, is_angular=False):
                if len(values) < 2:
                    return values[0][1] if values else None  
                view_indices = [v[0] for v in values]
                vals = np.array([v[1] for v in values])
                
                if is_angular:
                    vals = vals % (2 * np.pi)
                    diff = np.abs(vals[:, None] - vals)
                    angular_dists = np.minimum(diff, 2 * np.pi - diff)
                    avg_dists = np.mean(angular_dists, axis=1)
                else:
                    if not is_3d:
                        vals = vals.reshape(-1, 1)
                    pairwise_dists = squareform(pdist(vals, metric='euclidean'))
                    avg_dists = np.mean(pairwise_dists, axis=1)
                
                min_dist_idx = np.argmin(avg_dists)
                return values[min_dist_idx][1] 
            def momentum_yaw_fusion(current_yaw, prev_yaw, momentum=0.7):
                if current_yaw is None:
                    return prev_yaw if prev_yaw is not None else None
                if prev_yaw is None:
                    return current_yaw
                current_yaw = current_yaw % (2 * np.pi)
                prev_yaw = prev_yaw % (2 * np.pi)
                sin_momentum = momentum * np.sin(prev_yaw) + (1 - momentum) * np.sin(current_yaw)
                cos_momentum = momentum * np.cos(prev_yaw) + (1 - momentum) * np.cos(current_yaw)
                return np.arctan2(sin_momentum, cos_momentum) % (2 * np.pi)

            # Process widths, heights, lengths, yaw3d
            if valid_widths:
                self.width = select_lowest_avg_dist(valid_widths, "width")
            if valid_heights:
                self.height = select_lowest_avg_dist(valid_heights, "height")
            if valid_lengths:
                self.length = select_lowest_avg_dist(valid_lengths, "length")
            if valid_yaw3ds:
                # Select the most consistent yaw across views
                current_yaw = select_lowest_avg_dist(valid_yaw3ds, "yaw3d", is_angular=True)
                
                # Compute angular difference
                if self.yaw3d is not None:
                    current_yaw = current_yaw % (2 * np.pi)
                    prev_yaw = self.yaw3d % (2 * np.pi)
                    yaw_diff = np.minimum(
                        np.abs(current_yaw - prev_yaw),
                        2 * np.pi - np.abs(current_yaw - prev_yaw)
                    )
                else:
                    yaw_diff = 0.0
                
                # Define yaw change threshold (e.g., 60 degrees = pi/4 radians)
                yaw_change_threshold = np.pi / 4
                
                # Get class_id for momentum adjustment
                class_ids = [self.track2ds[v].class_id for v in valid_views if self.track2ds[v].class_id is not None]
                current_class_id = max(class_ids, default=-1) if class_ids else -1
                base_momentum = 0.8 if current_class_id in [4, 5] else 0.7
                
                # Increase momentum if yaw change is too large
                momentum = min(0.95, base_momentum + 0.2) if yaw_diff > yaw_change_threshold else base_momentum
                
                # Apply momentum fusion
                self.yaw3d = momentum_yaw_fusion(current_yaw, self.yaw3d, momentum)
            # Process 3D center
            if valid_bbox_centers:
                selected_center = select_lowest_avg_dist(valid_bbox_centers, "bbox_center3d", is_3d=True)
                if selected_center is not None:
                    self.bbox_center3d = np.array(selected_center)
            self.bbox_center3d = self.refine_3d_center(valid_views)
    
            # center, width, height, length, yaw = self.optimize_3d_bbox(valid_views)
            # self.width = width
            # self.height = height
            # self.length = length
        # If no valid views with sufficient keypoint scores, retain previous values (do nothing)
        # Check class_id from valid views
        # class_ids = [self.track2ds[v].class_id for v in valid_views if self.track2ds[v].class_id is not None]
        # # Use the most recent class_id (from valid views with age_bbox == 0)
        # current_class_id = max(class_ids, default=-1) if class_ids else -1

        # # For class_id 2,3 compute output_cord based on bbox center
        # if current_class_id in [2,3]:
        #     for v in valid_views:
        #         if self.age_bbox[v] == 0 and self.bbox_mv[v][-1] > 0.7:  # Ensure bbox is valid
        #             bbox = self.bbox_mv[v]
        #             # # Compute bbox center in 2D
        #             # if current_class_id ==2:
        #             #     bbox_center_2d = np.array([(bbox[0] + bbox[2]) / 2, (((bbox[1] + bbox[3]) / 2) + bbox[3])/2])
        #             # else:
        #             bbox_center_2d = np.array([(bbox[0] + bbox[2]) / 2, ((bbox[1] + bbox[3]) / 2) ])
        #             # Apply homography transformation using homo_inv
        #             center_homo = self.cameras[v].homo_inv @ np.array([bbox_center_2d[0], bbox_center_2d[1], 1])
        #             if abs(center_homo[-1]) > 1e-5:  # 
        #                 center_homo = center_homo[:-1] / center_homo[-1]
        #                 self.output_cord = np.concatenate((center_homo, [1]))
        #                 self.update_trajectory()
        #                 return self.output_cord
        #     # If no valid bbox center found, fall back to default (bbox bottom point)
        #     bottom_points = []
        #     for v in valid_views:
        #         bbox = self.bbox_mv[v]
        #         bp = self.cameras[v].homo_inv @ np.array([(bbox[0] + bbox[2]) / 2, bbox[3], 1])
        #         if abs(bp[-1]) > 1e-5:
        #             bp = bp[:-1] / bp[-1]
        #             bottom_points.append(bp)
        #     if bottom_points:
        #         bottom_points = np.array(bottom_points).reshape(-1, 2)
        #         self.output_cord = np.concatenate((np.mean(bottom_points, axis=0), [1]))
        #         self.update_trajectory()

        #         return self.output_cord
        #     self.output_cord = np.zeros(3)
        #     return self.output_cord
        
 
        # 3D kp output
        for comb in self.output_priority:
            if all(self.age_3D[comb] == 0):
                self.output_cord = np.concatenate((np.mean(self.keypoints_3d[comb, :2], axis=0), [3]))
                self.update_trajectory()
                return self.output_cord 
        
        # If no 3D kp comb, choose single-view feet 
        feet_idxs = self.output_priority[-1]
        for v in self.valid_views:
            if all(self.keypoints_mv[v][feet_idxs, -1] > self.keypoint_thrd) and all(self.age_2D[v][feet_idxs] == 0):
                feet_pos = np.mean(self.keypoints_mv[v][feet_idxs, :2], axis=0)
                feet_homo = self.cameras[v].homo_feet_inv @ np.array([feet_pos[0], feet_pos[1], 1])
                feet_homo = feet_homo[:-1] / feet_homo[-1]
                self.output_cord = np.concatenate((feet_homo, [2]))
                self.update_trajectory()
                return self.output_cord

        # If no single-view feet, then choose bbox bottom point
        bottom_points = []
        for v in self.valid_views:
            bbox = self.bbox_mv[v]
            bp = self.cameras[v].homo_inv @ np.array([(bbox[2] + bbox[0]) / 2, bbox[3], 1])
            bp = bp[:-1] / bp[-1]
            if bbox[3] > 1078:
                bottom_points.append(bp)
                continue
            self.output_cord = np.concatenate((bp, [1]))
            self.update_trajectory()
            return self.output_cord

        bottom_points = np.array(bottom_points).reshape(-1, 2)
        self.output_cord = np.concatenate((np.mean(bottom_points, axis=0), [1]))
        
        # # Smooth output_cord using Kalman filter
        # self.center_kalman.predict()
        # if self.output_cord[2] > 0:
        #     if np.all(self.center_kalman.get_state()[:2] == 0):
        #         self.center_kalman.init_state(self.output_cord[:2], high_covariance=True)
            
        #     if self.output_cord[2] == 3:
        #         R = np.diag([0.1, 0.1])
        #     elif self.output_cord[2] == 2:
        #         R = np.diag([1.0, 1.0])
        #     else:
        #         R = np.diag([10.0, 10.0])
        #     self.center_kalman.update(self.output_cord[:2], R)
        #     if np.all(self.bbox_center3d != 0):
        #         R_bbox = np.diag([1.0, 1.0])
        #         self.center_kalman.update(self.bbox_center3d[:2], R_bbox)
        #     self.output_cord[:2] = self.center_kalman.get_state()[:2]
        # self.output_cord = self.bbox_center3d
        self.update_trajectory()
        return self.output_cord

    
    def single_view_init(self, detection_sample, id):
        # If initialized only with 1 view 
        self.state = TrackState.Unconfirmed
        self.confirm_time_left = 2
        cam_id = detection_sample.cam_id
        self.valid_views.append(cam_id)

        track2d = self.track2ds[cam_id]
        track2d.init_with_det(detection_sample)
        self.bbox_mv[cam_id] = detection_sample.bbox
        self.keypoints_mv[cam_id] = detection_sample.keypoints_2d
        self.age_2D[cam_id][detection_sample.keypoints_2d[:, -1] > self.keypoint_thrd] = 0
        self.age_bbox[cam_id] = 0
        self.bbox_kalman[cam_id].update(detection_sample.bbox[:4].copy())
        self.feat_bank[0] = track2d.reid_feat
        self.feat_count += 1
        self.id = id
        self.update_age = 0
        self.dura_bbox[cam_id] = 1
        # Store 3D parameters
        self.width_mv[cam_id] = detection_sample.width
        self.height_mv[cam_id] = detection_sample.height
        self.length_mv[cam_id] = detection_sample.length
        self.yaw3d_mv[cam_id] = detection_sample.yaw3d
        self.bbox_center3d_mv[cam_id] = detection_sample.bbox_center3d
        self.get_output()  # Update fused dimensions
    def triangulation(self, detection_sample_list):
        keypoints_mv = np.zeros((self.num_cam, self.num_keypoints, 3))
        keypoints_3d = np.zeros((self.num_keypoints, 4))
        age_2D = np.ones((self.num_cam, self.num_keypoints)) * np.inf
        age_3D = np.ones((self.num_keypoints)) * np.inf

        for sample in detection_sample_list:
            keypoints_mv[sample.cam_id] = sample.keypoints_2d

        valid_joint_mask = (keypoints_mv[:, :, 2] > (self.keypoint_thrd if sample.class_id not in [0,4,5] else self.keypoint_thrd +0.1 ))

        for j_idx in range(self.num_keypoints):
            if np.sum(valid_joint_mask[:, j_idx]) < 2:
                joint_3d = np.zeros(4)
            else:
                A = np.zeros((2 * self.num_keypoints, 4))
                for v_idx in range(self.num_cam):
                    if valid_joint_mask[v_idx, j_idx]:
                        A[2 * v_idx, :] = keypoints_mv[v_idx, j_idx, 2] * (
                            keypoints_mv[v_idx, j_idx, 0] * self.cameras[v_idx].project_mat[2, :] - 
                            self.cameras[v_idx].project_mat[0, :]
                        )
                        A[2 * v_idx + 1, :] = keypoints_mv[v_idx, j_idx, 2] * (
                            keypoints_mv[v_idx, j_idx, 1] * self.cameras[v_idx].project_mat[2, :] - 
                            self.cameras[v_idx].project_mat[1, :]
                        )
                u, sigma, vt = np.linalg.svd(A)
                joint_3d = vt[-1] / (vt[-1][-1] + 1e-5)
                age_3D[j_idx] = 0
            keypoints_3d[j_idx] = joint_3d
            age_2D[valid_joint_mask[:, j_idx]] = 0

        return keypoints_3d, keypoints_mv, age_3D, age_2D

    def multi_view_init(self, detection_sample_list, id):
        self.state = TrackState.Confirmed
        self.keypoints_3d, self.keypoints_mv, self.age_3D, self.age_2D = self.triangulation(detection_sample_list)

        for sample in detection_sample_list:
            cam_id = sample.cam_id
            self.valid_views.append(cam_id)
            track2d = self.track2ds[cam_id]
            track2d.init_with_det(sample)
            self.bbox_mv[cam_id] = sample.bbox
            self.bbox_kalman[cam_id].update(sample.bbox[:4].copy())
            self.width_mv[cam_id] = sample.width
            self.height_mv[cam_id] = sample.height
            self.length_mv[cam_id] = sample.length
            self.yaw3d_mv[cam_id] = sample.yaw3d
            self.bbox_center3d_mv[cam_id] = sample.bbox_center3d

            if all(sample.keypoints_2d[self.upper_body, -1] > 0.5) and sample.bbox[4] > 0.8 and \
               np.sum(self.iou_mv[cam_id] > 0.15) < 1 and np.sum(self.ovr_mv[cam_id] > 0.3) < 2:
                reid_thrd= self.thred_reid_robot if sample.class_id !=0 else self.thred_reid
                if self.feat_count:
                    bank = self.feat_bank[:self.feat_count]
                    sim = bank @ sample.reid_feat
                    if np.max(sim) < (reid_thrd ):
                        self.feat_bank[self.feat_count % self.bank_size] = sample.reid_feat
                        self.feat_count += 1
                else:
                    self.feat_bank[0] = track2d.reid_feat
                    self.feat_count += 1

            self.age_bbox[cam_id] = 0
            self.dura_bbox[cam_id] = 1

        self.update_age = 0
        self.id = id
        self.iou_mv = [0 for i in range(self.num_cam)]
        self.valid_views = sorted(self.valid_views)
        self.get_output()
    def single_view_2D_update(self, v, sample, iou, ovr, ovr_tgt, avail_idx):
        if np.sum(iou > 0.5) >= 2 or np.sum(ovr_tgt > 0.5) >= 2:
            self.oc_state[v] = True
            oc_idx = avail_idx[np.where((iou > 0.5) | (ovr_tgt > 0.5))]
            self.oc_idx[v] = list(set(self.oc_idx[v] + [i for i in oc_idx if i != self.id]))
            
        valid_joints = sample.keypoints_2d[:, -1] > self.keypoint_thrd
        self.keypoints_mv[v][valid_joints] = sample.keypoints_2d[valid_joints]
        self.age_2D[v][valid_joints] = 0
        self.bbox_mv[v] = sample.bbox
        self.age_bbox[v] = 0
        self.track2ds[v].init_with_det(sample)
        self.bbox_kalman[v].update(sample.bbox[:4].copy())
        self.dura_bbox[v] += 1
        self.iou_mv[v] = iou
        self.ovr_mv[v] = ovr
        self.ovr_tgt_mv[v] = ovr_tgt

        # Update 3D parameters

        if self._is_valid_sample_joint_pairs(sample.keypoints_2d):
            self.width_mv[v] = sample.width
            self.height_mv[v] = sample.height
            self.length_mv[v] = sample.length
            self.yaw3d_mv[v] = sample.yaw3d
            self.bbox_center3d_mv[v] = sample.bbox_center3d
        self.update_trajectory()
    def multi_view_3D_update(self, avail_tracks):
        valid_views = [v for v in range(self.num_cam) if (self.age_bbox[v] == 0)]
        if self.feat_count >= self.bank_size:
            bank = self.feat_bank
        else:
            bank = self.feat_bank[:self.feat_count % self.bank_size]

        for v in valid_views:
            if self.oc_state[v] and np.sum(self.iou_mv[v] > 0.15) < 2 and np.sum(self.ovr_tgt_mv[v] > 0.3) < 2 and self.bbox_mv[v][-1] > 0.74:
                if self.feat_count == 0:
                    self.oc_state[v] = False
                    self.oc_idx[v] = []
                    continue
                self_sim = np.max((self.track2ds[v].reid_feat @ bank.T))
                self.oc_state[v] = False
                oc_tracks = []
                if self_sim > 0.5:
                    self.oc_idx[v] = []
                    continue
                for t_id, track in enumerate(avail_tracks):
                    if track.id in self.oc_idx[v]:
                        oc_tracks.append(track)
                if len(oc_tracks) == 0:
                    self.oc_idx[v] = []
                    continue
                reid_sim = np.zeros(len(oc_tracks))
                for t_id, track in enumerate(oc_tracks):
                    if track.feat_count == 0:
                        continue
                    if track.feat_count >= track.bank_size:
                        oc_bank = track.feat_bank
                    else:
                        oc_bank = track.feat_bank[:track.feat_count % track.bank_size]
                    reid_sim[t_id] = np.max(self.track2ds[v].reid_feat @ oc_bank.T)
                max_idx = np.argmax(reid_sim)
                self.oc_idx[v] = []
                if reid_sim[max_idx] > self_sim and reid_sim[max_idx] > 0.5:
                    self.switch_view(oc_tracks[max_idx], v)

        valid_joint_mask = (self.keypoints_mv[:, :, 2] > self.keypoint_thrd) & (self.age_2D == 0)
        corr_v = []
        # New: Triangulate 3D keypoints for each valid view pair and select the most consistent
        # keypoints_3d_candidates = []
        # view_pairs = []
        # avg_dists = []

        # from itertools import combinations
        # for view_pair in combinations(valid_views, 2):
        #     v1, v2 = view_pair
        #     if not (np.any(valid_joint_mask[v1]) and np.any(valid_joint_mask[v2])):
        #         continue
        #     keypoints_3d_temp = np.zeros((self.num_keypoints, 4))
        #     age_3d_temp = np.ones(self.num_keypoints) * np.inf
        #     for j_idx in range(self.num_keypoints):
        #         if valid_joint_mask[v1, j_idx] and valid_joint_mask[v2, j_idx]:
        #             A = np.zeros((4, 4))
        #             A[0, :] = self.keypoints_mv[v1, j_idx, 2] * (
        #                 self.keypoints_mv[v1, j_idx, 0] * self.cameras[v1].project_mat[2, :] -
        #                 self.cameras[v1].project_mat[0, :]
        #             )
        #             A[1, :] = self.keypoints_mv[v1, j_idx, 2] * (
        #                 self.keypoints_mv[v1, j_idx, 1] * self.cameras[v1].project_mat[2, :] -
        #                 self.cameras[v1].project_mat[1, :]
        #             )
        #             A[2, :] = self.keypoints_mv[v2, j_idx, 2] * (
        #                 self.keypoints_mv[v2, j_idx, 0] * self.cameras[v2].project_mat[2, :] -
        #                 self.cameras[v2].project_mat[0, :]
        #             )
        #             A[3, :] = self.keypoints_mv[v2, j_idx, 2] * (
        #                 self.keypoints_mv[v2, j_idx, 1] * self.cameras[v2].project_mat[2, :] -
        #                 self.cameras[v2].project_mat[1, :]
        #             )
        #             u, sigma, vt = np.linalg.svd(A)
        #             joint_3d = vt[-1] / (vt[-1][-1] + 1e-5)
        #             # Validate 3D point (e.g., check z-coordinate)
        #             if joint_3d[2] < -1 or joint_3d[2] > 2.5 or (j_idx in self.feet_idx and (joint_3d[2] < -1 or joint_3d[2] > 1)):
        #                 continue
        #             keypoints_3d_temp[j_idx] = joint_3d
        #             age_3d_temp[j_idx] = 0
        #     if np.any(age_3d_temp == 0):  # Only consider pairs with valid 3D points
        #         keypoints_3d_candidates.append(keypoints_3d_temp)
        #         view_pairs.append(view_pair)
        #         # Compute consistency metric (average pairwise distance among valid keypoints)
        #         valid_joints = age_3d_temp == 0
        #         if np.sum(valid_joints) > 0:
        #             points_3d = keypoints_3d_temp[valid_joints, :3]
        #             pairwise_dists = squareform(pdist(points_3d, metric='euclidean'))
        #             avg_dist = np.mean(pairwise_dists) if pairwise_dists.size > 0 else np.inf
        #             avg_dists.append(avg_dist)
        #         else:
        #             avg_dists.append(np.inf)

        # # Select the most consistent 3D keypoints
        # if keypoints_3d_candidates:
        #     best_idx = np.argmin(avg_dists)
        #     self.keypoints_3d = keypoints_3d_candidates[best_idx]
        #     self.age_3D = np.ones(self.num_keypoints) * np.inf
        #     self.age_3D[self.keypoints_3d[:, -1] != 0] = 0
        # else:
        for j_idx in range(self.num_keypoints):
            if np.sum(valid_joint_mask[:, j_idx]) < 2:
                joint_3d = np.zeros(4)
                continue
            else:
                A = np.zeros((2 * self.num_keypoints, 4))
                for v_idx in range(self.num_cam):
                    if valid_joint_mask[v_idx, j_idx]:
                        A[2 * v_idx, :] = self.keypoints_mv[v_idx, j_idx, 2] * (
                            self.keypoints_mv[v_idx, j_idx, 0] * self.cameras[v_idx].project_mat[2, :] - 
                            self.cameras[v_idx].project_mat[0, :]
                        )
                        A[2 * v_idx + 1, :] = self.keypoints_mv[v_idx, j_idx, 2] * (
                            self.keypoints_mv[v_idx, j_idx, 1] * self.cameras[v_idx].project_mat[2, :] - 
                            self.cameras[v_idx].project_mat[1, :]
                        )
                u, sigma, vt = np.linalg.svd(A)
                joint_3d = vt[-1] / vt[-1][-1]
                if (joint_3d[2] < -1 or joint_3d[2] > 2.5) or (j_idx in self.feet_idx and (joint_3d[2] < -1 or joint_3d[2] > 1)):
                    if np.min(self.dura_bbox[self.age_bbox == 0]) >= 10:
                        continue
                    v_cand = [v for v in range(self.num_cam) if (self.dura_bbox[v] == np.min(self.dura_bbox[self.age_bbox == 0]))]
                    for v in v_cand:
                        if valid_joint_mask[v, j_idx]:
                            self.age_bbox[v] = np.inf
                            self.dura_bbox[v] = 0
                            self.keypoints_mv[v] = 0
                            self.age_2D[v] = np.inf
                            valid_joint_mask[v] = 0
                            corr_v.append(v)
                            break
                    continue
                self.age_3D[j_idx] = np.min(self.age_2D[valid_joint_mask[:, j_idx], j_idx])
            self.keypoints_3d[j_idx] = joint_3d

        valid_views = [v for v in range(self.num_cam) if (self.age_bbox[v] == 0 and (not v in corr_v))]
        self.update_age = 0

        for v in valid_views:
            if self.feat_count >= self.bank_size:
                bank = self.feat_bank
            else:
                bank = self.feat_bank[:self.feat_count]
            sample = self.track2ds[v]
            reid_thrd= self.thred_reid_robot if sample.class_id !=0 else self.thred_reid
            if all(sample.keypoints_2d[self.upper_body, -1] > 0.5) and sample.bbox[4] > 0.74 and \
               np.sum(self.iou_mv[v] > 0.15) < 2 and np.sum(self.ovr_mv[v] > 0.3) < 2:
                if self.feat_count == 0:
                    self.feat_bank[0] = sample.reid_feat
                    self.feat_count += 1
                else:
                    sim = bank @ sample.reid_feat
                    if np.max(sim) < (reid_thrd ):
                        self.feat_bank[self.feat_count % self.bank_size] = sample.reid_feat
                        self.feat_count += 1

        if self.state == TrackState.Unconfirmed:
            if any(self.bbox_mv[self.age_bbox == 0][:, -1] > 0.74):
                self.confirm_time_left -= 1
                if self.confirm_time_left <= 0:
                    self.state = TrackState.Confirmed
        
        self.iou_mv = [0 for i in range(self.num_cam)]
        self.ovr_mv = [0 for i in range(self.num_cam)]
        self.get_output()  # Update fused dimensions
        return corr_v

    def CalcTargetRays(self, v):
        if self.age_bbox[v] > 1:
            return self.unit
        cam = self.cameras[v]
        return aic_cpp.compute_joints_rays(self.keypoints_mv[v], cam.project_inv, cam.pos)


def calcRays_sv(keypoints_2d, cam):
    joints_h = np.vstack((keypoints_2d[:, :-1].T, np.ones((1, keypoints_2d.shape[0]))))
    joints_rays = cam.project_inv @ joints_h
    joints_rays /= joints_rays[-1]
    joints_rays = joints_rays[:-1]
    joints_rays -= np.repeat(cam.pos.reshape(3, 1), keypoints_2d.shape[0], axis=1)
    joints_rays_norm = joints_rays / (np.linalg.norm(joints_rays, axis=0) + 1e-5)
    joints_rays_norm = joints_rays_norm.T
    return joints_rays_norm


class PoseTracker:
    def __init__(self, cameras):
        self.cameras = cameras
        self.num_cam = len(cameras)
        self.tracks = []
        self.reid_thrd = 0.5
        self.num_keypoints = 17
        self.decay_weight = 0.5
        self.thred_p2l_3d = 0.3
        self.thred_2d = 0.3
        self.thred_epi = 0.2
        self.thred_homo = 1.5
        self.thred_bbox = 0.4
        self.keypoint_thrd = 0.7
        self.glpk_bip = GLPKSolver(min_affinity=-1e5)
        self.main_joints = np.array([5,6,11,12,13,14,15,16])
        self.bank_size = 30
        self.thred_reid = 0.5
        self.upper_body = np.array([5,6,11,12])
        self.thred_3diou = np.float16(2.0)

    # def compute_frechet_aff(self, detection_sample_list_mv, avail_tracks):
    #         aff_mv = []
    #         n_track = len(avail_tracks)
    #         for v in range(self.num_cam):
    #             aff_sv = np.zeros((len(detection_sample_list_mv[v]), n_track))
    #             for s_id, sample in enumerate(detection_sample_list_mv[v]):
    #                 for t_id, track in enumerate(avail_tracks):
    #                     if len(track.history) >= 5 and sample.class_id != 0 or int(track.track2ds[v].class_id) != 0: # Minimum history length
    #                         traj_track = np.array(track.history)
    #                         N = min(len(traj_track)+1, 10)
    #                         traj_detect = np.vstack((traj_track, track.output_cord[:2]))[-N:]
                            
    #                         # Clean NaNs
    #                         traj_track_clean = traj_track[~np.isnan(traj_track).any(axis=1)]
    #                         traj_detect_clean = traj_detect[~np.isnan(traj_detect).any(axis=1)]
                            
    #                         if len(traj_track_clean) < 2 or len(traj_detect_clean) < 2:
    #                             distance = 9999999
    #                         else:
    #                             # detection uses last known track position
    #                             # Convert to list of tuples for Frechet function
    #                             P = [tuple(p) for p in traj_track_clean]
    #                             Q = [tuple(q) for q in traj_detect_clean]
    #                             distance = discrete_frechet(P, Q)
    #                             # print("DEBUG distance", distance)
                                
    #                         aff_sv[s_id, t_id] = np.exp(-distance)
    #                     else:
    #                         aff_sv[s_id, t_id] = 0
    #             aff_mv.append(aff_sv)
    #         return aff_mv
    def compute_frechet_aff(self, detection_sample_list_mv, avail_tracks):
        aff_mv, w_mv = [], []
        n_track = len(avail_tracks)

        for v in range(self.num_cam):
            aff_sv = np.zeros((len(detection_sample_list_mv[v]), n_track), dtype=float)
            w_sv = np.zeros((len(detection_sample_list_mv[v]), n_track), dtype=float)

            for s_id, sample in enumerate(detection_sample_list_mv[v]):
                for t_id, track in enumerate(avail_tracks):
                    if len(track.history) >= 10:
                        traj_track = np.array(list(track.history))
                        # Validate traj_track and output_cord
                        if np.any(np.isnan(traj_track)) or np.any(np.isinf(traj_track)) or \
                        np.any(np.isnan(track.output_cord[:2])) or np.any(np.isinf(track.output_cord[:2])):
                            aff_sv[s_id, t_id] = 0
                            w_sv[s_id, t_id] = 0
                            continue
                        
                        # Ensure proper shape for vstack
                        output_cord = track.output_cord[:2].reshape(1, 2)
                        traj_detect = np.vstack((traj_track, output_cord))[-10:]
                        
                        # Remove NaNs
                        mask = ~np.isnan(traj_track).any(axis=1) & ~np.isnan(traj_detect).any(axis=1)
                        traj_track_clean = traj_track[mask]
                        traj_detect_clean = traj_detect[mask]
                        
                        if len(traj_track_clean) >= 2 and len(traj_detect_clean) >= 2:
                            # Compute max Euclidean distance as approximation
                            distance = np.max(np.linalg.norm(traj_track_clean - traj_detect_clean, axis=1))
                            if np.isinf(distance):
                                aff_sv[s_id, t_id] = 0
                            else:
                                aff_sv[s_id, t_id] = np.exp(-distance)
                            w_sv[s_id, t_id] = 1
                        else:
                            aff_sv[s_id, t_id] = 0
                            w_sv[s_id, t_id] = 0
                    
                    else:
                        aff_sv[s_id, t_id] = 0
                        w_sv[s_id, t_id] = 0

            aff_mv.append(aff_sv)
            w_mv.append(w_sv)

        # Ensure aff_mv and w_mv are free of NaN/inf
        aff_mv = [np.nan_to_num(aff, nan=0.0, posinf=0.0, neginf=0.0) for aff in aff_mv]
        w_mv = [np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0) for w in w_mv]
        
        return aff_mv, w_mv
    
    
    def compute_3diou_aff(self, detection_sample_list_mv, avail_tracks):
        """
        Compute 3D IoU affinity matrix between detections and tracks for each view.
        Incorporates yaw-sensitive IoU with penalty for differences >= 60 degrees.
        
        Args:
            detection_sample_list_mv (list): List of detection samples per view.
            avail_tracks (list): List of available tracks.
        
        Returns:
            list: Affinity matrices [aff_sv_0, aff_sv_1, ..., aff_sv_n] for each view.
        """
        aff_3diou = []
        for v in range(self.num_cam):
            samples = detection_sample_list_mv[v]
            aff_sv = np.zeros((len(samples), len(avail_tracks)))
            for s_id, sample in enumerate(samples):
                joints_s = sample.keypoints_2d
                main_valid_s = np.all(joints_s[self.main_joints, -1] > self.keypoint_thrd)
                if sample.bbox[-1] < self.thred_bbox and not main_valid_s:
                    continue
                if (sample.width is None or sample.length is None or sample.height is None or 
                    sample.bbox_center3d is None or np.all(sample.bbox_center3d == 0)):
                    if not np.all(sample.bbox == 0):
                        sample_center = np.array([(sample.bbox[0] + sample.bbox[2]) / 2, 
                                                (sample.bbox[1] + sample.bbox[3]) / 2])
                        for t_id, track in enumerate(avail_tracks):
                            if not np.all(track.bbox_mv[v] == 0):
                                track_center = np.array([(track.bbox_mv[v][0] + track.bbox_mv[v][2]) / 2, 
                                                        (track.bbox_mv[v][1] + track.bbox_mv[v][3]) / 2])
                                dist = np.linalg.norm(sample_center - track_center)
                                aff_sv[s_id, t_id] = np.exp(-dist / 100)
                    continue
                for t_id, track in enumerate(avail_tracks):
                    if (np.all(track.bbox_center3d == 0) or track.width is None or 
                        track.length is None or track.height is None):
                        continue

                    box1 = {
                        'center': sample.bbox_center3d,
                        'dims': np.array([sample.width, sample.length ,sample.height]),
                        'yaw': sample.yaw3d if sample.yaw3d is not None and not np.isnan(sample.yaw3d) else 0.0
                    }
                    box2 = {
                        'center': track.bbox_center3d,
                        'dims': np.array([track.width , track.length, track.height]),
                        'yaw': track.yaw3d if track.yaw3d is not None and not np.isnan(track.yaw3d) else 0.0
                    }
                    # Use yaw-sensitive IoU
                    # iou = compute_3d_iou_fast_approximation(
                    #     box1, 
                    #     box2, 
                    #     yaw_threshold=np.pi/2, 
                    #     yaw_penalty_factor=2
                    # )
                    iou = compute_3d_iou_convex_hull(box1,box2)
                    if iou > 0.3:
                        aff_sv[s_id, t_id] = iou
                    # else:
                    #     dist = np.linalg.norm(sample.bbox_center3d - track.bbox_center3d)
                    #     aff_sv[s_id, t_id] = np.exp(-dist/5)

            aff_3diou.append(aff_sv)
        return aff_3diou
    
    def compute_reid_aff(self, detection_sample_list_mv, avail_tracks):
        reid_sim_mv = []
        reid_weight = []
        n_track = len(avail_tracks)

        for v in range(self.num_cam):
            reid_sim_sv = np.zeros((len(detection_sample_list_mv[v]), n_track))
            reid_weight_sv = np.zeros((len(detection_sample_list_mv[v]), n_track)) + 1e-5
    
            for s_id, sample in enumerate(detection_sample_list_mv[v]):
                if sample.bbox[-1] < 0.74:
                    continue
                if sample.class_id !=0 :
                    continue
                for t_id, track in enumerate(avail_tracks):
                    if not len(track.track2ds[v].state):
                        continue
                    if (Track2DState.Occluded not in track.track2ds[v].state) and (Track2DState.Missing not in track.track2ds[v].state):
                        continue 
                    reid_sim = track.feat_bank @ sample.reid_feat
                    reid_sim = reid_sim[reid_sim > 0]
                    if reid_sim.size:
                        reid_sim_sv[s_id, t_id] = np.max(reid_sim)
                        reid_weight_sv[s_id, t_id] = 1
                        reid_sim_sv[s_id, t_id] -= self.reid_thrd

            reid_sim_mv.append(reid_sim_sv)
            reid_weight.append(reid_weight_sv)
        return reid_sim_mv, reid_weight

    def compute_reid_aff_for_init(self, detection_sample_list_mv):
        """
        Compute ReID affinity matrix for detections across all camera views.
        Returns aff_reid (det_num x det_num) and reid_weight (det_num x det_num).
        """
        det_count = [len(detection_sample_list_mv[v]) for v in range(self.num_cam)]
        det_all_count = [0] + [sum(det_count[:v+1]) for v in range(self.num_cam)]
        det_num = det_all_count[-1]
        aff_reid = np.ones((det_num, det_num)) * (-10000)
        reid_weight = np.zeros((det_num, det_num))

        for vi in range(self.num_cam):
            samples_vi = detection_sample_list_mv[vi]
            for vj in range(vi, self.num_cam):
                if vi == vj:
                    continue
                samples_vj = detection_sample_list_mv[vj]
                reid_sim_temp = np.zeros((det_count[vi], det_count[vj]))
                for a in range(det_count[vi]):
                    sample_a = samples_vi[a]
                    for b in range(det_count[vj]):
                        sample_b = samples_vj[b]
                        # Check if ReID features are valid (non-zero)
                        if np.linalg.norm(sample_a.reid_feat) > 1e-5 and np.linalg.norm(sample_b.reid_feat) > 1e-5:
                            reid_sim_temp[a, b] = sample_a.reid_feat @ sample_b.reid_feat
                            reid_weight[det_all_count[vi] + a, det_all_count[vj] + b] = 1
                        else:
                            reid_sim_temp[a, b] = 0  # Penalize invalid ReID features
                aff_reid[det_all_count[vi]:det_all_count[vi + 1], det_all_count[vj]:det_all_count[vj + 1]] = reid_sim_temp
                aff_reid[det_all_count[vj]:det_all_count[vj + 1], det_all_count[vi]:det_all_count[vi + 1]] = reid_sim_temp.T

        return aff_reid, reid_weight
    def compute_3dkp_aff(self, detection_sample_list_mv, avail_tracks):
        aff_mv = []
        n_track = len(avail_tracks)

        for v in range(self.num_cam):
            aff_sv = np.zeros((len(detection_sample_list_mv[v]), n_track))
            cam = self.cameras[v]
            for s_id, sample in enumerate(detection_sample_list_mv[v]):
                joints_h = np.vstack((sample.keypoints_2d[:, :-1].T, np.ones((1, self.num_keypoints))))
                joints_rays = cam.project_inv @ joints_h
                joints_rays /= joints_rays[-1]
                joints_rays = joints_rays[:-1]
                joints_rays -= np.repeat(cam.pos.reshape(3, 1), self.num_keypoints, axis=1)
                joints_rays = joints_rays / (np.linalg.norm(joints_rays, axis=0) + 1e-5)
                joints_rays = joints_rays.T
                for t_id, track in enumerate(avail_tracks):
                    aff = np.zeros(self.num_keypoints)
                    kp_3d = track.keypoints_3d
                    k_idx = np.where(sample.keypoints_2d[:, -1] < self.keypoint_thrd)[0]
                    aff[k_idx] = Point2LineDist(kp_3d[k_idx, :-1], cam.pos, joints_rays[k_idx])
                    valid = (sample.keypoints_2d[:, -1] > self.keypoint_thrd) * (kp_3d[:, -1] > 0)
                    aff = 1 - aff / self.thred_p2l_3d
                    aff = aff * sample.keypoints_2d[:, -1] * np.exp(-track.age_3D)
                    aff_sv[s_id, t_id] = np.sum(aff) / (np.sum(valid * np.exp(-track.age_3D)) + 1e-5)
            aff_mv.append(aff_sv)
        return aff_mv
    
    def compute_2dkp_aff(self, detection_sample_list_mv, avail_tracks):
        aff_mv = []
        n_track = len(avail_tracks)

        for v in range(self.num_cam):
            aff_sv = np.zeros((len(detection_sample_list_mv[v]), n_track))
            for s_id, sample in enumerate(detection_sample_list_mv[v]):
                joints_s = sample.keypoints_2d
                for t_id, track in enumerate(avail_tracks):
                    joints_t = track.keypoints_mv[v]
                    dist = np.linalg.norm(joints_t[:, :-1] - joints_s[:, :-1], axis=1)
                    aff = 1 - dist / (self.thred_2d * (np.linalg.norm(track.bbox_mv[v][2:4] - track.bbox_mv[v][:2]) + 1e-5))
                    valid = (joints_t[:, -1] > self.keypoint_thrd) * (joints_s[:, -1] > self.keypoint_thrd)
                    aff = aff * valid * np.exp(-track.age_2D[v])
                    aff_sv[s_id, t_id] = np.sum(aff) / (np.sum(valid * np.exp(-track.age_2D[v])) + 1e-5)
            aff_mv.append(aff_sv)
        return aff_mv    
    
    def compute_epi_homo_aff(self, detection_sample_list_mv, avail_tracks):
        aff_mv = []
        aff_homo = [] 
        n_track = len(avail_tracks)
        mv_rays = self.CalcJointRays(detection_sample_list_mv)
        age_2D_thr = 1
        feet_idxs = [15, 16]

        for v in range(self.num_cam):
            pos = self.cameras[v].pos
            aff_sv = np.zeros((len(detection_sample_list_mv[v]), n_track))
            cam = self.cameras[v]
            sv_rays = mv_rays[v]
            aff_homo_sv = np.zeros((len(detection_sample_list_mv[v]), n_track))

            for s_id, sample in enumerate(detection_sample_list_mv[v]):
                joints_s = sample.keypoints_2d
                feet_valid_s = np.all(joints_s[feet_idxs, -1] > self.keypoint_thrd)
                feet_s = aic_cpp.compute_feet_s(joints_s, feet_idxs, cam.homo_feet_inv)
                box_pos_s = aic_cpp.compute_box_pos_s(sample.bbox, cam.homo_inv)
                box_valid_s = True

                for t_id, track in enumerate(avail_tracks):
                    joints_t = track.keypoints_mv
                    aff_sv[s_id, t_id], aff_homo_sv[s_id, t_id] = aic_cpp.loop_t_homo_full(
                        joints_t,
                        joints_s,
                        track.age_bbox,
                        track.age_2D,
                        feet_s,
                        feet_valid_s,
                        v,
                        self.thred_epi,
                        self.thred_homo,
                        self.keypoint_thrd,
                        age_2D_thr,
                        sv_rays[s_id],
                        self.cameras,
                        box_pos_s,
                        box_valid_s,
                        track.bbox_mv)
                    continue
                    aff_ss = []
                    aff_homo_ss = []
                    if feet_valid_s:
                        feet_valid_t = (joints_t[:, feet_idxs[0], -1] > self.keypoint_thrd) & (joints_t[:, feet_idxs[1], -1] > self.keypoint_thrd)

                    valid = (joints_t[:, :, -1] > self.keypoint_thrd) & (joints_s[:, -1] > self.keypoint_thrd)
                    for vj in range(self.num_cam):
                        if v == vj or track.age_bbox[vj] >= 2:
                            continue
                        pos_j = self.cameras[vj].pos
                        track_rays_sv = track.CalcTargetRays(vj)
                        aff_temp = aic_cpp.epipolar_3d_score_norm(pos, sv_rays[s_id], pos_j, track_rays_sv, self.thred_epi)
                        _aff_ss = aic_cpp.aff_sum(aff_temp, valid[vj], track.age_2D[vj], 1)
                        if _aff_ss != 0:
                            aff_ss.append(_aff_ss)
                        if feet_valid_s and feet_valid_t[vj]:
                            _aff_homo_ss = aic_cpp.compute_feet_distance(
                                joints_t[vj], feet_idxs, self.cameras[vj].homo_feet_inv, feet_s, self.thred_homo)
                            aff_homo_ss.append(_aff_homo_ss)
                    aff_homo_sv[s_id, t_id] = sum(aff_homo_ss) / (len(aff_homo_ss) + 1e-5)
                    aff_sv[s_id, t_id] = sum(aff_ss) / (len(aff_ss) + 1e-5)
            aff_homo.append(aff_homo_sv)
            aff_mv.append(aff_sv)
        return aff_mv, aff_homo

    def compute_bboxiou_aff(self, detection_sample_list_mv, avail_tracks):
        aff_mv = []
        iou_mv = []
        ovr_det_mv = []
        ovr_tgt_mv = []
        n_track = len(avail_tracks)
        for v in range(self.num_cam):
            iou = np.zeros((len(detection_sample_list_mv[v]), n_track))
            ovr_det = np.zeros((len(detection_sample_list_mv[v]), len(detection_sample_list_mv[v])))
            if iou.size == 0:
                aff_mv.append(iou)
                iou_mv.append(iou)
                ovr_det_mv.append(ovr_det)
                ovr_tgt_mv.append(iou)
                continue
            detection_bboxes = np.stack([detection.bbox for detection in detection_sample_list_mv[v]])[:, :5]
            multi_mean = np.stack([
                track.bbox_kalman[v].mean.copy() if track.bbox_kalman[v].mean is not None else np.array([1, 1, 1, 1, 0, 0, 0, 0])
                for track in avail_tracks
            ])
            multi_covariance = np.stack([
                track.bbox_kalman[v].covariance.copy() if track.bbox_kalman[v].covariance is not None else np.eye(8)
                for track in avail_tracks
            ])
            multi_mean, multi_covariance = avail_tracks[0].bbox_kalman[v].multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                if avail_tracks[i].bbox_kalman[v].mean is not None:
                    avail_tracks[i].bbox_kalman[v].mean = mean
                    avail_tracks[i].bbox_kalman[v].covariance = cov
            score = detection_bboxes[:, -1]
            detection_bboxes = detection_bboxes[:, :4]
            track_bboxes = self.xyah2ltrb(multi_mean[:, :4].copy())
            for i in range(len(track_bboxes)):
                if avail_tracks[i].bbox_kalman[v].mean is None:
                    track_bboxes[i] = avail_tracks[i].bbox_mv[v][:4]
            iou = ious(detection_bboxes.copy(), track_bboxes.copy())
            iou[np.isnan(iou)] = 0
            age = np.array([track.age_bbox[v] for track in avail_tracks])
            ovr = aic_cpp.bbox_overlap_rate(detection_bboxes.copy(), track_bboxes.copy())
            ovr_tgt_mv.append(ovr * (age <= 15))
            ovr_det = aic_cpp.bbox_overlap_rate(detection_bboxes.copy(), detection_bboxes.copy())
            ovr_det_mv.append(ovr_det)
            iou_mv.append(iou * (age <= 15))
            iou = (((iou - 0.5) * (age <= 15)).T * score).T
            aff_mv.append(iou)
        return aff_mv, iou_mv, ovr_det_mv, ovr_tgt_mv

    def CalcJointRays(self, detection_sample_list_mv):
        mv_rays = []
        for v in range(self.num_cam):
            cam = self.cameras[v]
            sv_rays = []
            n_detect = len(detection_sample_list_mv[v])
            sample_sv = detection_sample_list_mv[v]
            for s_id, sample in enumerate(sample_sv):
                joints_h = np.vstack((sample.keypoints_2d[:, :-1].T, np.ones((1, self.num_keypoints))))
                joints_rays = cam.project_inv @ joints_h
                joints_rays /= joints_rays[-1]
                joints_rays = joints_rays[:-1]
                joints_rays -= np.repeat(cam.pos.reshape(3, 1), self.num_keypoints, axis=1)
                joints_rays_norm = joints_rays / (np.linalg.norm(joints_rays, axis=0) + 1e-5)
                joints_rays_norm = joints_rays_norm.T
                sv_rays.append(joints_rays_norm)
            mv_rays.append(sv_rays)
        return mv_rays
    
    def match_with_miss_tracks(self, new_track, miss_tracks):
        if len(miss_tracks) == 0:
            self.tracks.append(new_track)
            return
        
        reid_sim = np.zeros(len(miss_tracks))
        frechet_sim = np.zeros(len(miss_tracks))
        pos_sim = np.zeros(len(miss_tracks))
        for t_id, track in enumerate(miss_tracks):
            print(track.id, new_track.id)
            if track.feat_count == 0 or new_track.feat_count == 0:
                continue
            if track.feat_count >= self.bank_size:
                bank = track.feat_bank
            else:
                bank = track.feat_bank[:track.feat_count % self.bank_size]
            new_bank = new_track.feat_bank[:new_track.feat_count%self.bank_size]
            
            reid_sim[t_id] = np.max(new_bank @ bank.T)
            if not np.all(new_track.bbox_center3d == 0) and not np.all(track.bbox_center3d == 0):
                dist = np.linalg.norm(new_track.bbox_center3d - track.bbox_center3d)
                pos_sim[t_id] = np.exp(-dist/10)
            # Frechet distance similarity
            # if len(track.history) >= 5 and len(new_track.history) >= 5:
            #     N = min(len(track.history), 10)
            #     P = [tuple(p) for p in track.history[-N:]]
            #     Q = [tuple(q) for q in new_track.history[-N:]]
            #     frechet_distance = discrete_frechet(P, Q)
            #     frechet_sim[t_id] = np.exp(-frechet_distance)
            # else:
            #     frechet_sim[t_id] = 0.0
        combined_sim = 0.95 * reid_sim + 0.05 * pos_sim #+ 0.2*frechet_sim
        t_id = np.argmax(combined_sim)
        print(combined_sim, reid_sim, frechet_sim,pos_sim)
        # t_id = np.argmax(reid_sim)
        threshold = 0.5 - 0.1 * (len(new_track.valid_views) > 1) - 0.05 * (miss_tracks[t_id].update_age / 30)
        threshold = max(0.35, threshold)  # Ensure threshold doesn't go too low
        
        if combined_sim[t_id] > threshold:
            print("Match miss")
            miss_tracks[t_id].reactivate(new_track)
        else:
            self.tracks.append(new_track)

    
    def merge_duplicate_tracks(self):
        tracks_to_keep = []
        for i, track_i in enumerate(self.tracks):
            if track_i.state >= TrackState.Missing:
                continue
            keep = True
            for j, track_j in enumerate(self.tracks[i + 1:], start=i + 1):
                if track_j.state >= TrackState.Missing:
                    continue
                # Compute similarity
                sim = 0  # Default to no similarity
                if track_i.feat_count > 0 and track_j.feat_count > 0:
                    # Handle feature bank slicing
                    bank_i = track_i.feat_bank[:min(track_i.feat_count, self.bank_size)]
                    bank_j = track_j.feat_bank[:min(track_j.feat_count, self.bank_size)]
                    if bank_i.size > 0 and bank_j.size > 0:  # Ensure non-empty arrays
                        sim = np.max(bank_i @ bank_j.T)
                # Fallback to 3D position similarity
                if sim == 0 and np.any(track_i.age_3D < 3) and np.any(track_j.age_3D < 3):
                    pos_diff = np.linalg.norm(track_i.output_cord[:2] - track_j.output_cord[:2])
                    sim = max(0, 1 - pos_diff / self.thred_homo)  # Ensure non-negative
                # Merge if similarity is high and positions are close
                if sim > 0.6 and np.linalg.norm(track_i.output_cord[:2] - track_j.output_cord[:2]) < self.thred_homo:
                    track_i.reactivate(track_j)
                    track_j.state = TrackState.Deleted
                    keep = False
                    # print(f"Merged track {track_j.id} into {track_i.id}: sim={sim:.3f}")
                # else:
                #     print(f"Skipped merging track {track_i.id} vs {track_j.id}: sim={sim:.3f}")
            if keep:
                tracks_to_keep.append(track_i)
        self.tracks = tracks_to_keep
    
    def target_init(self, detection_sample_list_mv, miss_tracks, iou_det_mv, ovr_det_mv, ovr_tgt_mv):
        cam_idx_map = []
        det_count = []
        det_all_count = [0]
        for v in range(self.num_cam):
            det_count.append(len(detection_sample_list_mv[v]))
            det_all_count.append(det_all_count[-1] + det_count[-1])
            cam_idx_map += [v] * det_count[-1]
        
        if det_all_count[-1] == 0:
            return self.tracks

        det_num = det_all_count[-1]
        aff_homo = np.ones((det_num, det_num)) * (-10000)
        aff_epi = np.ones((det_num, det_num)) * (-10000)
        mv_rays = self.CalcJointRays(detection_sample_list_mv)
        feet_idxs = [15, 16]
        # aff_reid, reid_weight = self.compute_reid_aff_for_init(detection_sample_list_mv)
        for vi in range(self.num_cam):
            samples_vi = detection_sample_list_mv[vi]
            pos_i = self.cameras[vi].pos
            for vj in range(vi, self.num_cam):
                if vi == vj:
                    continue
                else:
                    pos_j = self.cameras[vj].pos
                    samples_vj = detection_sample_list_mv[vj]
                    aff_temp = np.zeros((det_count[vi], det_count[vj]))
                    reid_sim_temp = np.zeros((det_count[vi], det_count[vj]))
                    aff_homo_temp = np.zeros((det_count[vi], det_count[vj]))
                    for a in range(det_count[vi]):
                        sample_a = samples_vi[a]
                        feet_valid_a = np.all(sample_a.keypoints_2d[feet_idxs, -1] > self.keypoint_thrd)
                        if feet_valid_a:
                            feet_a = np.mean(sample_a.keypoints_2d[feet_idxs, :-1], axis=0)
                            feet_a = self.cameras[vi].homo_feet_inv @ np.array([feet_a[0], feet_a[1], 1])
                            feet_a = feet_a[:-1] / feet_a[-1]
                        else:
                            feet_a = np.array([(sample_a.bbox[0] + sample_a.bbox[2]) / 2, sample_a.bbox[3]])
                            feet_a = self.cameras[vi].homo_inv @ np.array([feet_a[0], feet_a[1], 1])
                            feet_a = feet_a[:-1] / feet_a[-1]
                        feet_valid_a = True
                        for b in range(det_count[vj]):
                            sample_b = samples_vj[b]
                            aff = np.zeros(self.num_keypoints)
                            valid_kp = (sample_a.keypoints_2d[:, -1] > self.keypoint_thrd) & (sample_b.keypoints_2d[:, -1] > self.keypoint_thrd)
                            j_id = np.where(valid_kp)[0]
                            aff[j_id] = aic_cpp.epipolar_3d_score_norm(pos_i, mv_rays[vi][a][j_id, :], pos_j, mv_rays[vj][b][j_id, :], self.thred_epi)
                            if feet_valid_a and np.all(sample_b.keypoints_2d[feet_idxs, -1] > self.keypoint_thrd):
                                feet_b = np.mean(sample_b.keypoints_2d[feet_idxs, :-1], axis=0)
                                feet_b = self.cameras[vj].homo_feet_inv @ np.array([feet_b[0], feet_b[1], 1])
                                feet_b = feet_b[:-1] / feet_b[-1]
                                aff_homo_temp[a, b] = 1 - np.linalg.norm(feet_b - feet_a) / self.thred_homo
                            else:
                                feet_b = np.array([(sample_b.bbox[0] + sample_b.bbox[2]) / 2, sample_b.bbox[3]])
                                feet_b = self.cameras[vj].homo_feet_inv @ np.array([feet_b[0], feet_b[1], 1])
                                feet_b = feet_b[:-1] / feet_b[-1]
                                aff_homo_temp[a, b] = 1 - np.linalg.norm(feet_b - feet_a) / self.thred_homo
                            aff_temp[a, b] = np.sum(aff * sample_a.keypoints_2d[:, -1] * sample_b.keypoints_2d[:, -1]) / (
                                np.sum(valid_kp * sample_a.keypoints_2d[:, -1] * sample_b.keypoints_2d[:, -1]) + 1e-5
                            )
                            reid_sim_temp[a, b] = (sample_a.reid_feat @ sample_b.reid_feat)
                    aff_epi[det_all_count[vi]:det_all_count[vi + 1], det_all_count[vj]:det_all_count[vj + 1]] = aff_temp
                    aff_homo[det_all_count[vi]:det_all_count[vi + 1], det_all_count[vj]:det_all_count[vj + 1]] = aff_homo_temp

        aff_final = 2 * aff_epi + aff_homo #+ 3 * aff_reid * reid_weight
        # aff_final =( 2 * aff_epi + aff_homo + 3 * aff_reid * reid_weight)/(1+reid_weight)
        aff_final[aff_final < -1000] = -np.inf
        clusters, sol_matrix = self.glpk_bip.solve(aff_final, True)

        for cluster in clusters:
            if len(cluster) == 1:
                view_list, number_list = find_view_for_cluster(cluster, det_all_count)
                det = detection_sample_list_mv[view_list[0]][number_list[0]]
                is_robot = det.class_id != 0
                valid_bbox = det.bbox[-1] > self.thred_bbox

                valid_pose = True if is_robot else all(det.keypoints_2d[self.main_joints, -1] > 0.5)
                isolated_IOU = np.sum(iou_det_mv[view_list[0]][number_list[0]] > 0.15) < 1
                isolated_detect = np.sum(ovr_det_mv[view_list[0]][number_list[0]] > 0.3) < 2

                if valid_bbox and valid_pose and isolated_IOU and isolated_detect:
                    new_track = PoseTrack(self.cameras)
                    new_track.single_view_init(det, id=len(self.tracks) + 1)
                    self.match_with_miss_tracks(new_track, miss_tracks)
            else:
                view_list, number_list = find_view_for_cluster(cluster, det_all_count)
                sample_list = [detection_sample_list_mv[view_list[idx]][number_list[idx]] for idx in range(len(view_list))]
                for i, sample in enumerate(sample_list):
                    is_robot_mv = sample.class_id != 0
                    valid_bbox_mv = sample.bbox[-1] > self.thred_bbox

                    valid_pose_mv = True if is_robot_mv else all(sample.keypoints_2d[self.main_joints, -1] > 0.5)
                    isolated_IOU_mv = np.sum(iou_det_mv[view_list[i]][number_list[i]] > 0.15) < 1
                    isolated_detect_mv = np.sum(ovr_det_mv[view_list[i]][number_list[i]] > 0.3) < 2
                    if valid_pose_mv and valid_bbox_mv and isolated_IOU_mv and isolated_detect_mv:
                        new_track = PoseTrack(self.cameras)
                        for j in range(len(view_list)):
                            new_track.iou_mv[view_list[j]] = iou_det_mv[view_list[j]][number_list[j]]
                            new_track.ovr_mv[view_list[j]] = ovr_det_mv[view_list[j]][number_list[j]]
                            new_track.ovr_tgt_mv[view_list[j]] = ovr_tgt_mv[view_list[j]][number_list[j]]
                        new_track.multi_view_init(sample_list, id=len(self.tracks) + 1)
                        self.match_with_miss_tracks(new_track, miss_tracks)
                        break
    
    def xyah2ltrb(self, ret):
        ret[..., 2] *= ret[..., 3]
        ret[..., :2] -= ret[..., 2:] / 2
        ret[..., 2:] += ret[..., :2]
        return ret

    def mv_update_wo_pred(self, detection_sample_list_mv, frame_id=None):
        um_iou_det_mv = []
        um_ovr_det_mv = []
        um_ovr_tgt_mv = []
        a_epi = 2
        a_box = 5 #7
        a_homo = 1 #2
        a_reid = 3
        a_3diou = 1
        a_frechet = 1
        for track in self.tracks:
            track.valid_views = []
        avail_tracks = [track for track in self.tracks if track.state < TrackState.Missing]
        avail_idx = np.array([track.id for track in avail_tracks])

        aff_reid, reid_weight = self.compute_reid_aff(detection_sample_list_mv, avail_tracks)
        aff_epi, aff_homo = self.compute_epi_homo_aff(detection_sample_list_mv, avail_tracks)
        aff_box, iou_mv, ovr_det_mv, ovr_tgt_mv = self.compute_bboxiou_aff(detection_sample_list_mv, avail_tracks)
        # print(f'aff_reid :{aff_reid}')
        # print(f'aff_epi :{aff_epi}')
        # print(f'aff_homo :{aff_homo}')
        # print(f'aff_box :{aff_box}')

        aff_3diou = self.compute_3diou_aff(detection_sample_list_mv, avail_tracks)
        # print(aff_3diou)
        aff_traj, traj_weight = self.compute_frechet_aff(detection_sample_list_mv, avail_tracks)
        updated_tracks = set()
        unmatched_det = []
        match_result = []

        for v in range(self.num_cam):
            iou_sv = iou_mv[v]
            ovr_det_sv = ovr_det_mv[v]
            ovr_tgt_sv = ovr_tgt_mv[v]
            matched_det_sv = set()

            aff_epi[v][aff_epi[v] < (-a_box + 0.5 * a_box)] = -a_box + 0.5 * a_box
            aff_homo[v][aff_homo[v] < (-a_box + 0.5 * a_box)] = -a_box + 0.5 * a_box
            aff_3diou[v][aff_3diou[v] < 0] = 0

            norm = a_epi * (aff_epi[v] != 0).astype(float) + a_box * (aff_box[v] != 0).astype(float) + a_homo * (aff_homo[v] != 0).astype(float) + a_3diou * (aff_3diou[v] != 0).astype(float)  + a_frechet * (aff_traj[v] != 0).astype(float) 
            aff_final = (a_epi * aff_epi[v] + a_box * aff_box[v] + a_homo * aff_homo[v] + 
                         aff_reid[v] * a_reid * reid_weight[v] + a_3diou * aff_3diou[v] + a_frechet*aff_traj[v]*traj_weight[v]) / (
                1 + reid_weight[v]
            )
            # for s_id, sample in enumerate(detection_sample_list_mv[v]):
            #     penalty = 0
            #     if 0.5 < sample.bbox[-1] <self.thred_bbox:
            #         penalty = 0.2
            #     elif sample.bbox[-1] <0.5:
            #         penalty = 0.3
            #     aff_final[s_id] -= penalty
            # norm = a_epi *(aff_epi[v]!=0).astype(float) + a_box *(aff_box[v]!=0).astype(float)+ a_homo *(aff_homo[v]!=0).astype(float)
            # aff_final = (a_epi*aff_epi[v] + a_box*aff_box[v] + a_homo*aff_homo[v]+aff_reid[v]*a_reid*reid_weight[v])/(1+reid_weight[v])

            idx = np.where(norm > 0)
            aff_final[idx] -= (a_box - norm[idx]) * 0.1
            sample_list_sv = detection_sample_list_mv[v]
            aff_final[aff_final < 0] = 0
            # print(aff_final)
            row_idxs, col_idxs = linear_sum_assignment(-aff_final)
            match_result.append((row_idxs, col_idxs))
            if iou_sv.size:
                colmax = iou_sv.max(0)
                argcolmax = iou_sv.argmax(0)
                occlusion_row = set()
                for i in range(iou_sv.shape[1]):
                    if i not in col_idxs:
                        # print(colmax[i])
                        if colmax[i] > 0.5:
                            state = Track2DState.Occluded
                            occlusion_row.add(argcolmax[i])
                        else:
                            state = Track2DState.Missing
                        if len(avail_tracks[i].track2ds[v].state) == 10:
                            avail_tracks[i].track2ds[v].state.pop()
                        avail_tracks[i].track2ds[v].state = [state] + avail_tracks[i].track2ds[v].state
            elif len(iou_sv) == 0:
                for i in range(iou_sv.shape[1]):
                    if len(avail_tracks[i].track2ds[v].state) == 10:
                        avail_tracks[i].track2ds[v].state.pop()
                    avail_tracks[i].track2ds[v].state = [Track2DState.Missing] + avail_tracks[i].track2ds[v].state

            for row, col in zip(row_idxs, col_idxs):

                if row in occlusion_row:
                    state = Track2DState.Occluded
                    if len(avail_tracks[col].track2ds[v].state) == 10:
                        avail_tracks[col].track2ds[v].state.pop()
                    avail_tracks[col].track2ds[v].state = [state] + avail_tracks[col].track2ds[v].state
                if aff_final[row, col] <= 0:
                    continue
                iou = iou_sv[row]
                ovr = ovr_det_sv[row]
                ovr_tgt = ovr_tgt_sv[row]
                avail_tracks[col].single_view_2D_update(v, sample_list_sv[row], iou, ovr, ovr_tgt, avail_idx)
                updated_tracks.add(col)
                matched_det_sv.add(row)
                    
            unmatched_det_sv = list(set(range(len(sample_list_sv))) - matched_det_sv)
            unmatched_sv = [sample_list_sv[u] for u in unmatched_det_sv]
            unmatched_det.append(unmatched_sv)
            unmatched_iou_sv = iou_sv[unmatched_det_sv]
            um_iou_det_mv.append(unmatched_iou_sv)
            unmatched_ovr_det_sv = ovr_det_sv[unmatched_det_sv]
            um_ovr_det_mv.append(unmatched_ovr_det_sv)
            unmatched_ovr_tgt_sv = ovr_tgt_sv[unmatched_det_sv]
            um_ovr_tgt_mv.append(unmatched_ovr_tgt_sv)
    
        for t_id in updated_tracks:
            corr_v = avail_tracks[t_id].multi_view_3D_update(avail_tracks)

        for t_id, track in enumerate(avail_tracks):
            track.valid_views = [v for v in range(self.num_cam) if track.age_bbox[v] == 0]
            if track.state == TrackState.Unconfirmed and (not t_id in updated_tracks):
                track.state = TrackState.Deleted
            if track.update_age >= 60:
                track.state = TrackState.Missing
            # if track.update_age >= 90:
            #     track.state = TrackState.Deleted
            if track.state == TrackState.Confirmed:
                track.get_output()

        miss_tracks = [track for track in self.tracks if track.state == TrackState.Missing]
        if len(unmatched_det):
            self.target_init(unmatched_det, miss_tracks, um_iou_det_mv, um_ovr_det_mv, um_ovr_tgt_mv)
        # self.merge_duplicate_tracks()
        feat_cnts = []
        for track in self.tracks:
            for i in range(self.num_cam):
                if track.age_bbox[i] >= 60:
                    track.bbox_kalman[i] = KalmanFilter_box()
            track.age_2D[track.age_2D >= 3] = np.inf
            track.age_3D[track.age_3D >= 3] = np.inf
            track.age_bbox[track.age_bbox >= 60] = np.inf
            track.dura_bbox[track.age_bbox >= 60] = 0
            track.age_2D += 1
            track.age_3D += 1
            track.age_bbox += 1
            track.update_age += 1
            if track.state == TrackState.Confirmed:
                feat_cnts.append((track.id, track.feat_count))

    
    def output(self, frame_id):
        frame_results = []
        for track in self.tracks:
            if track.state == TrackState.Confirmed:
                if track.update_age == 1:
                    for v in track.valid_views:
                        bbox = track.bbox_mv[v]
                        record = np.array([[
                            self.cameras[v].idx_int,
                            track.id,
                            frame_id,
                            track.class_id,
                            bbox[0], bbox[1],
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1],
                            track.output_cord[0],
                            track.output_cord[1],
                            track.output_cord[2],
                            # track.width_mv[v],
                            # track.height_mv[v],
                            # track.length_mv[v],
                            # track.yaw3d_mv[v],
                            # track.bbox_center3d_mv[v][0],
                            # track.bbox_center3d_mv[v][1],
                            # track.bbox_center3d_mv[v][2],
                            track.width,
                            track.height,
                            track.length,
                            track.yaw3d,
                            track.bbox_center3d[0],
                            track.bbox_center3d[1],
                            track.bbox_center3d[2],
                        ]],dtype=np.float32)
                        frame_results.append(record)
        return frame_results