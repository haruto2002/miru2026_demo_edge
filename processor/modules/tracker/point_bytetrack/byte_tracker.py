from __future__ import annotations

import numpy as np

from processor.modules.tracker.point_bytetrack.basetrack import BaseTrack, TrackState
from processor.modules.tracker.point_bytetrack.kalman_filter import KalmanFilter
from processor.modules.tracker.point_bytetrack.matching import (
    euclidean_distance,
    linear_assignment,
    maha_distance,
)


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, point, score):
        self._point = np.asarray(point, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = float(score)
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        if not stracks:
            return
        multi_mean = np.asarray([st.mean for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
            multi_mean, multi_covariance
        )
        for i, st in enumerate(stracks):
            st.mean = multi_mean[i]
            st.covariance = multi_covariance[i]

    @staticmethod
    def multi_update(
        stracks,
        measurements: np.ndarray,
        scores,
        frame_id,
        reset_tracklet_len: bool = False,
    ):
        """Batched KF correction for matched tracks."""
        n = len(stracks)
        if n == 0:
            return
        means = np.asarray([st.mean for st in stracks], dtype=np.float64)
        covs = np.asarray([st.covariance for st in stracks], dtype=np.float64)
        meas = np.asarray(measurements, dtype=np.float64).reshape(n, 2)
        new_means, new_covs = STrack.shared_kalman.multi_update(means, covs, meas)
        for i, st in enumerate(stracks):
            st.mean = new_means[i]
            st.covariance = new_covs[i]
            st.score = float(scores[i])
            st.state = TrackState.Tracked
            st.is_activated = True
            st.frame_id = frame_id
            if st.kalman_filter is None:
                st.kalman_filter = STrack.shared_kalman
            if reset_tracklet_len:
                st.tracklet_len = 0
            else:
                st.tracklet_len += 1

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self._point)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    @property
    def point(self):
        """Current (x, y). Returns a view (no copy) for hot paths."""
        if self.mean is None:
            return self._point
        return self.mean[:2]

    def __repr__(self):
        return "OT_{}_({}-{})".format(self.track_id, self.start_frame, self.end_frame)


class PointBYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.match_thresh = args.match_thresh
        self.distance_metric = args.distance_metric

    def update(self, output_results):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        assert output_results.shape[1] == 3, (
            "output_results must be a 3D array (x, y, score)"
        )
        scores = output_results[:, 2]
        points = np.ascontiguousarray(output_results[:, :2], dtype=np.float32)

        remain_inds = scores > self.args.track_thresh
        inds_second = np.logical_and(scores > 0.1, scores <= self.args.track_thresh)

        points_keep = points[remain_inds]
        scores_keep = scores[remain_inds]
        points_second = points[inds_second]
        scores_second = scores[inds_second]

        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if track.is_activated:
                tracked_stracks.append(track)
            else:
                unconfirmed.append(track)

        """ Step 2: First association (arrays, no per-det objects) """
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        dists = maha_distance(
            strack_pool, points_keep, self.kalman_filter, metric=self.distance_metric
        )
        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self.match_thresh
        )

        to_update, upd_idx = [], []
        to_reactivate, rea_idx = [], []
        for itracked, idet in matches:
            track = strack_pool[itracked]
            if track.state == TrackState.Tracked:
                to_update.append(track)
                upd_idx.append(idet)
                activated_stracks.append(track)
            else:
                to_reactivate.append(track)
                rea_idx.append(idet)
                refind_stracks.append(track)
        if upd_idx:
            idx = np.asarray(upd_idx, dtype=np.int64)
            STrack.multi_update(
                to_update, points_keep[idx], scores_keep[idx], self.frame_id, False
            )
        if rea_idx:
            idx = np.asarray(rea_idx, dtype=np.int64)
            STrack.multi_update(
                to_reactivate, points_keep[idx], scores_keep[idx], self.frame_id, True
            )

        """ Step 3: Second association with low-score dets """
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = maha_distance(
            r_tracked_stracks,
            points_second,
            self.kalman_filter,
            metric=self.distance_metric,
        )
        matches, u_track, _ = linear_assignment(dists, self.match_thresh)
        to_update, upd_idx = [], []
        to_reactivate, rea_idx = [], []
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            if track.state == TrackState.Tracked:
                to_update.append(track)
                upd_idx.append(idet)
                activated_stracks.append(track)
            else:
                to_reactivate.append(track)
                rea_idx.append(idet)
                refind_stracks.append(track)
        if upd_idx:
            idx = np.asarray(upd_idx, dtype=np.int64)
            STrack.multi_update(
                to_update, points_second[idx], scores_second[idx], self.frame_id, False
            )
        if rea_idx:
            idx = np.asarray(rea_idx, dtype=np.int64)
            STrack.multi_update(
                to_reactivate,
                points_second[idx],
                scores_second[idx],
                self.frame_id,
                True,
            )

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """ Unconfirmed """
        if len(u_detection):
            left_idx = np.asarray(u_detection, dtype=np.int64)
            points_left = points_keep[left_idx]
            scores_left = scores_keep[left_idx]
        else:
            points_left = points_keep[:0]
            scores_left = scores_keep[:0]

        dists = maha_distance(
            unconfirmed,
            points_left,
            self.kalman_filter,
            metric=self.distance_metric,
        )
        matches, u_unconfirmed, u_detection2 = linear_assignment(
            dists, self.match_thresh
        )
        to_update, upd_idx = [], []
        for itracked, idet in matches:
            to_update.append(unconfirmed[itracked])
            upd_idx.append(idet)
        if upd_idx:
            idx = np.asarray(upd_idx, dtype=np.int64)
            STrack.multi_update(
                to_update, points_left[idx], scores_left[idx], self.frame_id, False
            )
            activated_stracks.extend(to_update)
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new tracks — only allocate STrack here """
        for inew in u_detection2:
            if scores_left[inew] < self.det_thresh:
                continue
            track = STrack(points_left[inew], scores_left[inew])
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        """ Step 5: Update state """
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, removed_stracks)
        self.removed_stracks = removed_stracks[-1000:]
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        return [t for t in self.tracked_stracks if t.is_activated]


def joint_stracks(tlista, tlistb):
    exists = {t.track_id for t in tlista}
    res = list(tlista)
    for t in tlistb:
        if t.track_id not in exists:
            exists.add(t.track_id)
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    remove_ids = {t.track_id for t in tlistb}
    return [t for t in tlista if t.track_id not in remove_ids]


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = euclidean_distance(stracksa, stracksb)
    pairs = np.where(pdist < 10.0)
    dupa, dupb = set(), set()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.add(q)
        else:
            dupa.add(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
