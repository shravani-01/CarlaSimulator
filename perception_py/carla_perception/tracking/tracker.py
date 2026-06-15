"""Multi-object tracking by IoU-based data association.

WHY THIS MODULE EXISTS
----------------------
A detector finds objects in ONE frame. But a driving system sees a video, and we
need to know that "the car at the left in frame 10" is the SAME car as "the car
slightly more left in frame 11". Assigning each object a persistent ID across
frames is *tracking*, and it's the foundation for anything motion-related: speed,
direction, trajectories, time-to-collision.

THE CORE PROBLEM: DATA ASSOCIATION
----------------------------------
Each frame we get a fresh, unordered list of detection boxes. We must decide
which new box continues which existing track. The simplest robust signal is
spatial overlap: between consecutive frames an object barely moves, so the new
box should heavily overlap its previous box. We measure overlap with IoU
(Intersection over Union) and greedily match the highest-overlap pairs.

This is a deliberately simple, readable tracker. Production stacks use
ByteTrack / BoT-SORT (which add motion models and appearance features), and we
can swap one in later behind this same `update()` interface.
"""

from __future__ import annotations

from dataclasses import dataclass

from carla_perception.detection.detector import Detection

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2)


def iou(box_a: Box, box_b: Box) -> float:
    """Intersection-over-Union of two axis-aligned boxes (0..1).

    1.0 = identical boxes, 0.0 = no overlap. This is our similarity measure for
    deciding whether two boxes refer to the same object.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Intersection rectangle.
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    """One tracked object, persisting across frames.

    Attributes:
        track_id: stable integer id for this object's whole lifetime.
        label:    most recent class label.
        xyxy:     most recent bounding box.
        hits:     how many frames we've matched this track (confidence it's real).
        age:      total frames since the track was created.
        time_since_update: frames since we last matched it (0 = seen this frame).
    """

    track_id: int
    label: str
    xyxy: Box
    hits: int = 1
    age: int = 0
    time_since_update: int = 0


class IoUTracker:
    """Greedy IoU tracker.

    Args:
        iou_threshold: minimum overlap to consider a detection a continuation of
            an existing track.
        max_age: how many frames a track may go unmatched before we delete it
            (lets a track survive a brief missed detection or occlusion).
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        """Advance the tracker by one frame and return the live tracks.

        Steps:
            1. Score every (track, detection) pair by IoU.
            2. Greedily accept the best pairs above threshold (each track and
               each detection used at most once).
            3. Update matched tracks; spawn tracks for unmatched detections;
               age-out tracks that went unmatched too long.
        """
        # Everyone ages by one frame; matching will reset the matched ones.
        for trk in self.tracks:
            trk.age += 1
            trk.time_since_update += 1

        # 1. Build candidate matches (iou, track_index, det_index).
        candidates: list[tuple[float, int, int]] = []
        for ti, trk in enumerate(self.tracks):
            for di, det in enumerate(detections):
                score = iou(trk.xyxy, det.xyxy)
                if score >= self.iou_threshold:
                    candidates.append((score, ti, di))

        # 2. Greedy assignment: highest IoU first, no reuse.
        candidates.sort(reverse=True)
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for _score, ti, di in candidates:
            if ti in matched_tracks or di in matched_dets:
                continue
            trk, det = self.tracks[ti], detections[di]
            trk.xyxy = det.xyxy
            trk.label = det.label
            trk.hits += 1
            trk.time_since_update = 0
            matched_tracks.add(ti)
            matched_dets.add(di)

        # 3a. Unmatched detections become new tracks.
        for di, det in enumerate(detections):
            if di not in matched_dets:
                self.tracks.append(
                    Track(track_id=self._next_id, label=det.label, xyxy=det.xyxy)
                )
                self._next_id += 1

        # 3b. Drop tracks that have been lost for too long.
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return the tracks we actually saw this frame.
        return [t for t in self.tracks if t.time_since_update == 0]
