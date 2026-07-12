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
    9: 16.919,
}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 4-dimensional state space

        x, y, vx, vy

    contains the bounding box center position (x, y) and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 2, 1.0

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y) with center position (x, y).

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (4 dimensional) and covariance matrix (4x4
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * 1.0,
            2 * self._std_weight_position * 1.0,
            10 * self._std_weight_velocity * 1.0,
            10 * self._std_weight_velocity * 1.0,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 4 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 4x4 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * 1.0,
            self._std_weight_position * 1.0,
        ]
        std_vel = [
            self._std_weight_velocity * 1.0,
            self._std_weight_velocity * 1.0,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # mean = np.dot(self._motion_mat, mean)
        mean = np.dot(mean, self._motion_mat.T)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (4 dimensional array).
        covariance : ndarray
            The state's covariance matrix (4x4 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * 1.0,
            self._std_weight_position * 1.0,
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx4 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx4x4 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        n = mean.shape[0]
        motion_cov = np.zeros((n, 4, 4), dtype=mean.dtype)
        sp = self._std_weight_position**2
        sv = self._std_weight_velocity**2
        motion_cov[:, 0, 0] = sp
        motion_cov[:, 1, 1] = sp
        motion_cov[:, 2, 2] = sv
        motion_cov[:, 3, 3] = sv

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step (single track)."""
        new_mean, new_cov = self.multi_update(
            mean[None], covariance[None], np.asarray(measurement, dtype=mean.dtype)[None]
        )
        return new_mean[0], new_cov[0]

    def multi_update(self, mean, covariance, measurement):
        """Batched Kalman correction for N tracks.

        Parameters
        ----------
        mean : ndarray, shape (N, 4)
        covariance : ndarray, shape (N, 4, 4)
        measurement : ndarray, shape (N, 2)

        Returns
        -------
        (ndarray, ndarray)
            new_mean (N, 4), new_covariance (N, 4, 4)
        """
        # Project to measurement space: Hx, HPH^T + R  (H picks x,y)
        # projected_mean: (N, 2)
        projected_mean = mean[:, :2]
        # projected_cov = P[:2,:2] + R
        r = self._std_weight_position**2
        s = covariance[:, :2, :2].copy()
        s[:, 0, 0] += r
        s[:, 1, 1] += r

        # 2x2 inverse of S (analytical; avoids N scipy.cho_* calls)
        a = s[:, 0, 0]
        b = s[:, 0, 1]
        c = s[:, 1, 0]
        d = s[:, 1, 1]
        det = a * d - b * c
        inv_s = np.empty_like(s)
        inv_s[:, 0, 0] = d / det
        inv_s[:, 0, 1] = -b / det
        inv_s[:, 1, 0] = -c / det
        inv_s[:, 1, 1] = a / det

        # PHT = P @ H.T = P[:, :, :2]  (N, 4, 2)
        pht = covariance[:, :, :2]
        # K = PHT @ inv(S)  (N, 4, 2)
        kalman_gain = np.matmul(pht, inv_s)

        innovation = measurement - projected_mean  # (N, 2)
        new_mean = mean + np.matmul(kalman_gain, innovation[..., None])[..., 0]
        # P = P - K @ S @ K^T
        new_covariance = covariance - np.matmul(
            np.matmul(kalman_gain, s), kalman_gain.transpose(0, 2, 1)
        )
        return new_mean, new_covariance

    def gating_distance(
        self, mean, covariance, measurements, only_position=False, metric="maha"
    ):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 2 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (4 dimensional).
        covariance : ndarray
            Covariance of the state distribution (4x4 dimensional).
        measurements : ndarray
            An Nx2 dimensional matrix of N measurements, each in
            format (x, y) where (x, y) is the bounding box center position.
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
        # only_position parameter is no longer effective (always using position only)
        # しかし、後方互換性のために残しています

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
            )
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError("invalid distance metric")
