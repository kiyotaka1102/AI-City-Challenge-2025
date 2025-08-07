# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter_box(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 10
        self._std_weight_velocity = 1. / 160
        self.mean=self.covariance=None
    def ltrb2xyah(self,ret):
        ret[...,2:]-=ret[...,:2]
        ret[...,:2] += ret[...,2:] / 2
        ret[...,2] /= ret[...,3]
        return ret
    

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        l,t,r,b=measurement
        measurement=self.ltrb2xyah(measurement)
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        self.mean = np.r_[mean_pos, mean_vel]
        '''if l<5:
            self.mean[5]+=measurement[3]*0.1
            self.mean[7]+=0.1
        if r>475:
            self.mean[5]-=measurement[3]*0.1
            self.mean[7]+=0.1
        if t<5:
            self.mean[6]+=measurement[3]*0.1
            self.mean[7]-=0.1
        if b>265:
            self.mean[6]-=measurement[3]*0.1
            self.mean[7]-=0.1'''
            
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        self.covariance = np.diag(np.square(std))
       

    '''def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        #mean = np.dot(self._motion_mat, mean)
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance'''

    def project(self):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * self.mean[3],
            self._std_weight_position * self.mean[3],
            1e-1,
            self._std_weight_position * self.mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, self.mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, self.covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parametersprint
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return mean,covariance

    def update(self, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        if self.mean is None:
            self.initiate(measurement)
        else:
            measurement=self.ltrb2xyah(measurement)
            projected_mean, projected_cov = self.project()

            chol_factor, lower = scipy.linalg.cho_factor(
                projected_cov, lower=True, check_finite=False)
            kalman_gain = scipy.linalg.cho_solve(
                (chol_factor, lower), np.dot(self.covariance, self._update_mat.T).T,
                check_finite=False).T * 0.5
            innovation = measurement - projected_mean
            innovation = innovation.clip(-measurement[3] * 0.5, measurement[3] * 0.5)

            self.mean = self.mean + np.dot(innovation, kalman_gain.T)
            self.covariance = self.covariance - np.linalg.multi_dot((
                kalman_gain, projected_cov, kalman_gain.T))
       

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')

class KalmanFilter_position3D:
    def __init__(self, dt=1.0 / 30.0, outlier_threshold=9.0, smoothing_factor=0.5):
        """
        Initialize a Kalman filter for 3D position tracking with constant velocity model and adaptive noise handling.
        Args:
            dt (float): Time step between frames (default: 1/30 for 30 FPS).
            outlier_threshold (float): Squared Mahalanobis distance threshold for outlier rejection.
            smoothing_factor (float): Weight for blending new measurement with prediction (0 to 1, higher = smoother).
        """
        self.state = np.zeros(6)  # [x, y, z, vx, vy, vz]
        self.covariance = np.eye(6) * 100

        # State transition matrix (constant velocity model)
        self.F = np.eye(6)
        self.F[:3, 3:] = np.eye(3) * dt

        # Measurement matrix (observe [x, y, z])
        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)

        # Process noise covariance (reduced for smoother tracking)
        q = 0.4
        Q_pos = q * (dt**3) / 3
        Q_vel = q * dt
        self.Q_base = np.diag([Q_pos, Q_pos, Q_pos, Q_vel, Q_vel, Q_vel])
        self.Q = self.Q_base.copy()

        # Measurement noise covariance (increased for smoother updates)
        r = 0.2
        self.R_base = np.diag([r, r, r * 2])
        self.R = self.R_base.copy()
        self.outlier_threshold = outlier_threshold
        self.velocity_damping = 0.9
        self.smoothing_factor = smoothing_factor

    def init_state(self, initial_position, high_covariance=True):
        """
        Initialize the state with a 3D position.
        Args:
            initial_position: numpy array of shape (3,) [x, y, z]
            high_covariance: bool, whether to use high initial uncertainty
        """
        initial_position = np.array(initial_position)
        if initial_position.shape != (3,):
            raise ValueError("initial_position must be a 3D vector")
        self.state[:3] = initial_position
        self.state[3:] = 0
        if high_covariance:
            self.covariance = np.eye(6) * 100
        else:
            self.covariance = np.eye(6) * 1.0
        self.Q = self.Q_base.copy()
        self.R = self.R_base.copy()

    def predict(self):
        """Predict the next state and covariance, applying velocity damping."""
        self.state[3:] *= self.velocity_damping
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q

    def update(self, measurement, R=None, num_views=1, avg_conf=0.5):
        """
        Update the state with a new 3D measurement, with robust outlier handling and smoothing.
        Args:
            measurement: numpy array of shape (3,) [x, y, z]
            R: measurement noise covariance, shape (3, 3), optional
            num_views: number of valid views contributing to the measurement
            avg_conf: average confidence of the contributing views
        """
        # Validate measurement
        measurement = np.array(measurement)
        if measurement.shape != (3,) or not np.all(np.isfinite(measurement)):
            raise ValueError("measurement must be a valid 3D vector")

        # Adjust process and measurement noise
        conf_scale = max(avg_conf, 0.1)
        view_scale = max(1.0 / num_views, 0.1)
        self.Q = self.Q_base * view_scale
        if R is not None:
            self.R = np.array(R)
            if self.R.shape != (3, 3):
                raise ValueError("R must be a 3x3 matrix")
        else:
            self.R = self.R_base * view_scale / conf_scale

        # Apply smoothing
        predicted_measurement = self.H @ self.state
        smoothed_measurement = (
            self.smoothing_factor * predicted_measurement +
            (1 - self.smoothing_factor) * measurement
        )

        # Compute innovation and innovation covariance
        innovation = smoothed_measurement - predicted_measurement
        S = self.H @ self.covariance @ self.H.T + self.R

        # Outlier detection
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S += np.eye(3) * 1e-6
            S_inv = np.linalg.inv(S)
        mahalanobis_dist_sq = innovation.T @ S_inv @ innovation

        if mahalanobis_dist_sq > self.outlier_threshold:
            self.R *= 100.0  # Reduce trust in measurement
            if mahalanobis_dist_sq > self.outlier_threshold * 4:
                return  # Skip extreme outliers

        # Compute Kalman gain
        try:
            K = self.covariance @ self.H.T @ S_inv
        except np.linalg.LinAlgError:
            S += np.eye(3) * 1e-6
            K = self.covariance @ self.H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.state = self.state + K @ innovation
        I_KH = np.eye(6) - K @ self.H
        self.covariance = I_KH @ self.covariance @ I_KH.T + K @ self.R @ K.T

    def get_state(self):
        """Return the current 3D position [x, y, z]."""
        return self.state[:3]

    def reset(self):
        """Reset the filter to initial state."""
        self.state = np.zeros(6)
        self.covariance = np.eye(6) * 100
        self.Q = self.Q_base.copy()
        self.R = self.R_base.copy()

# class KalmanFilter_position3D:
#     def __init__(self, dt=1.0 / 30.0):
#         """
#         Initialize a Kalman filter for 3D position tracking with constant velocity model.
#         Supports updates for x, y, z coordinates.
        
#         Args:
#             dt (float): Time step between frames (default: 1/30 for 30 FPS).
#         """
#         self.state = np.zeros(6)  # [x, y, z, vx, vy, vz]
#         self.covariance = np.diag([10, 10, 10, 10, 10, 10])  # Moderate initial uncertainty

#         # State transition matrix (constant velocity with damping)
#         damping = 0.99
#         self.F = np.eye(6)
#         self.F[:3, 3:] = np.eye(3) * dt
#         self.F[3:, 3:] = np.eye(3) * damping

#         # Measurement matrix (observe [x, y, z])
#         self.H = np.zeros((3, 6))
#         self.H[:3, :3] = np.eye(3)

#         # Process noise covariance
#         q_xy = 0.8
#         q_z = 0.01
#         Q_pos_xy = q_xy * (dt**3) / 3
#         Q_pos_z = q_z * (dt**3) / 3
#         Q_vel_xy = q_xy * dt
#         Q_vel_z = q_z * dt
#         self.Q = np.diag([Q_pos_xy, Q_pos_xy, Q_pos_z, Q_vel_xy, Q_vel_xy, Q_vel_z])

#         # Measurement noise covariance
#         r = 0.1
#         self.R = np.diag([r, r, 2*r])

#     def init_state(self, initial_position, high_covariance=True):
#         self.state[:3] = initial_position
#         self.state[3:] = 0
#         if high_covariance:
#             self.covariance = np.diag([10, 10, 10, 10, 10, 10])
#         else:
#             self.covariance = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

#     def predict(self):
#         self.state = self.F @ self.state
#         self.covariance = self.F @ self.covariance @ self.F.T + self.Q

#     def update(self, measurement, R=None, num_views=1, avg_conf=0.5):
#         if R is None:
#             r_scale = max(0.5 / (num_views * max(avg_conf, 0.1)), 0.1)
#             self.R = np.diag([r_scale, r_scale, r_scale])
#         else:
#             self.R = R
#         measurement = np.array(measurement)  # Use provided z
#         innovation = measurement - self.H @ self.state
#         S = self.H @ self.covariance @ self.H.T + self.R
#         try:
#             K = self.covariance @ self.H.T @ np.linalg.inv(S)
#         except np.linalg.LinAlgError:
#             S += np.eye(3) * 1e-6
#             K = self.covariance @ self.H.T @ np.linalg.inv(S)
#         self.state = self.state + K @ innovation
#         I_KH = np.eye(6) - K @ self.H
#         self.covariance = I_KH @ self.covariance @ I_KH.T + K @ self.R @ K.T

#     def get_state(self):
#         return self.state[:3]

#     def reset(self):
#         self.state = np.zeros(6)
#         self.covariance = np.diag([10, 10, 10, 10, 10, 10])