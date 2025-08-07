import numpy as np
def create_2d_heatmap(bbox, image_shape, sigma_scale=0.25):
    """Create a 2D Gaussian heatmap for a bounding box center."""
    center_2d = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    sigma = np.array([width, height]) * sigma_scale / max(bbox[-1], 0.1)  # Scale by confidence
    heatmap = np.zeros(image_shape)
    for y in range(image_shape[0]):
        for x in range(image_shape[1]):
            heatmap[y, x] = np.exp(-0.5 * np.sum(((np.array([x, y]) - center_2d) / sigma) ** 2))
    return heatmap / (np.sum(heatmap) + 1e-6)  # Normalize to probability density

def construct_3d_heatmap(views, bboxes_2d, cameras, grid_size=(50, 50, 50), bounds=(-10, 10)):
    """Construct a 3D heatmap from 2D heatmaps and camera poses."""
    x_range = np.linspace(bounds[0], bounds[1], grid_size[0])
    y_range = np.linspace(bounds[0], bounds[1], grid_size[1])
    z_range = np.linspace(-5, 5, grid_size[2])
    heatmap_3d = np.zeros(grid_size)
    
    for idx, (x, y, z) in enumerate(np.ndindex(grid_size)):
        point_3d = np.array([x_range[x], y_range[y], z_range[z]])
        prob = 0.0
        for v, bbox in zip(views, bboxes_2d):
            cam = cameras[v]
            proj_homo = cam.project_mat @ np.append(point_3d, 1)
            if abs(proj_homo[-1]) > 1e-5:
                proj_2d = proj_homo[:2] / proj_homo[-1]
                if 0 <= proj_2d[0] < cam.image_shape[1] and 0 <= proj_2d[1] < cam.image_shape[0]:
                    heatmap_2d = create_2d_heatmap(bbox, cam.image_shape)
                    prob += heatmap_2d[int(proj_2d[1]), int(proj_2d[0])]
        heatmap_3d[x, y, z] = prob / len(views)
    
    return heatmap_3d / (np.sum(heatmap_3d) + 1e-6), (x_range, y_range, z_range)

def sample_camera_poses(cameras, views, num_samples=100, translation_std=0.1, rotation_std=0.05):
    """Sample camera poses with Gaussian perturbations."""
    sampled_cameras = []
    for _ in range(num_samples):
        perturbed_cameras = {}
        for v in views:
            cam = cameras[v]
            # Perturb translation
            t_pert = np.random.normal(0, translation_std, 3)
            # Perturb rotation (simplified as small angle rotations)
            r_pert = np.random.normal(0, rotation_std, 3)
            rot_mat = cam.rotation @ rotation_matrix_from_euler(r_pert)
            perturbed_cameras[v] = Camera(
                project_mat=cam.project_mat @ np.vstack([np.hstack([rot_mat, t_pert[:, None]]), [0, 0, 0, 1]]),
                pos=cam.pos + t_pert,
                rotation=rot_mat,
                image_shape=cam.image_shape
            )
        sampled_cameras.append(perturbed_cameras)
    return sampled_cameras
def compute_posterior(views, bboxes_2d, cameras, sampled_cameras, grid_size=(50, 50, 50), bounds=(-10, 10)):
    """Compute posterior p(y|X^{2D}) and weighted 3D center."""
    likelihoods = []
    heatmaps_3d = []
    
    for sampled_cams in sampled_cameras:
        heatmap_3d, ranges = construct_3d_heatmap(views, bboxes_2d, sampled_cams, grid_size, bounds)
        reproj_error = 0.0
        for v, bbox in zip(views, bboxes_2d):
            cam = sampled_cams[v]
            center_2d = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            # Compute expected 2D heatmap from 3D heatmap
            heatmap_2d_reproj = np.zeros(cam.image_shape)
            for idx, (x, y, z) in enumerate(np.ndindex(grid_size)):
                point_3d = np.array([ranges[0][x], ranges[1][y], ranges[2][z]])
                proj_homo = cam.project_mat @ np.append(point_3d, 1)
                if abs(proj_homo[-1]) > 1e-5:
                    proj_2d = proj_homo[:2] / proj_homo[-1]
                    if 0 <= proj_2d[0] < cam.image_shape[1] and 0 <= proj_2d[1] < cam.image_shape[0]:
                        heatmap_2d_reproj[int(proj_2d[1]), int(proj_2d[0])] += heatmap_3d[x, y, z]
            heatmap_2d = create_2d_heatmap(bbox, cam.image_shape)
            reproj_error += np.sum((heatmap_2d - heatmap_2d_reproj) ** 2)
        likelihoods.append(np.exp(-reproj_error))
        heatmaps_3d.append(heatmap_3d)
    
    # Normalize likelihoods to get posterior
    likelihoods = np.array(likelihoods)
    posterior = likelihoods / (np.sum(likelihoods) + 1e-6)
    
    # Compute weighted 3D center
    weighted_heatmap = np.sum([p * h for p, h in zip(posterior, heatmaps_3d)], axis=0)
    x, y, z = np.array(np.unravel_index(np.argmax(weighted_heatmap), weighted_heatmap.shape))
    center_3d = np.array([ranges[0][x], ranges[1][y], ranges[2][z]])
    
    return center_3d, posterior