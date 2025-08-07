import numpy as np
import scipy.linalg
import json

class Camera:
    def __init__(self, cal_input):
        if isinstance(cal_input, str):
            with open(cal_input, 'r') as file:
                data = json.load(file)
        else:
            data = cal_input


        calib_data = data
        
        # Load matrices
        self.project_mat = np.array(calib_data["cameraMatrix"])
        self.homo_mat = np.array(calib_data["homography"])
        self.intrinsic_mat = np.array(calib_data["intrinsicMatrix"])
        self.extrinsic_mat = np.array(calib_data["extrinsicMatrix"])
        
        # Basic attributes
        self.scale_factor = calib_data.get("scaleFactor", 1.0)
        self.translation_to_global = calib_data.get("translationToGlobalCoordinates", [0, 0, 0])
        
        # Compute inverses
        self.homo_inv = np.linalg.inv(self.homo_mat)
        self.project_inv = scipy.linalg.pinv(self.project_mat)
        
        # Camera position
        self.pos = np.linalg.inv(self.project_mat[:, :-1]) @ -self.project_mat[:, -1]
        
        # Feet-level homography
        self.homo_feet = self.homo_mat.copy()
        self.homo_feet[:, -1] += self.project_mat[:, 2] * 0.15
        self.homo_feet_inv = np.linalg.inv(self.homo_feet)
        
        # Camera ID
        self.full_id = calib_data.get("id", "Camera")
        self.idx = self.full_id.split("_")[-1] if "_" in self.full_id else "00"
        self.idx_int = int(self.idx) if self.idx.isdigit() else 0

def cross(R, V):
    """Cross product helper function"""
    h = [R[1] * V[2] - R[2] * V[1],
         R[2] * V[0] - R[0] * V[2],
         R[0] * V[1] - R[1] * V[0]]
    return h

def Point2LineDist(p_3d, pos, ray):
    """Calculate distance from 3D point to line defined by position and ray"""
    return np.linalg.norm(np.cross(p_3d-pos, ray), axis=-1)

def Line2LineDist(pA, rayA, pB, rayB):
    """Calculate distance between two 3D lines"""
    if np.abs(np.dot(rayA, rayB)) > (1 - (1e-5)) * np.linalg.norm(rayA, axis=-1) * np.linalg.norm(rayB, axis=-1):  # quasi parallel
        return Point2LineDist(pA, pB, rayA)
    else:
        rayCP = np.cross(rayA, rayB)
        return np.abs((pA-pB).dot(rayCP / np.linalg.norm(rayCP, axis=-1), axis=-1))

def Line2LineDist_norm(pA, rayA, pB, rayB):
    """Normalized version of line-to-line distance calculation"""
    rayCP = np.cross(rayA, rayB, axis=-1)
    rayCP_norm = np.linalg.norm(rayCP, axis=-1) + 1e-6
    return np.abs(np.sum((pA-pB) * (rayCP / rayCP_norm[:, None]), -1))

def epipolar_3d_score(pA, rayA, pB, rayB, alpha_epi):
    """Calculate epipolar 3D score for triangulation quality"""
    dist = Line2LineDist(pA, rayA, pB, rayB)
    return 1 - dist/alpha_epi

def epipolar_3d_score_norm(pA, rayA, pB, rayB, alpha_epi):
    """Normalized version of epipolar 3D score"""
    dist = Line2LineDist_norm(pA, rayA, pB, rayB)
    return 1 - dist/alpha_epi
import aic_cpp
epipolar_3d_score_norm = aic_cpp.epipolar_3d_score_norm