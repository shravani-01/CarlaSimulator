"""Unit tests for the IoU tracker (no GPU / model needed - runs in CI)."""

from carla_perception.detection.detector import Detection
from carla_perception.tracking.tracker import IoUTracker, iou


def _det(label, xyxy):
    return Detection(cls_id=0, label=label, confidence=0.9, xyxy=xyxy)


def test_iou_identical_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes overlapping in a 5x10 strip:
    # inter=50, union=100+100-50=150 -> 1/3
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - (50 / 150)) < 1e-9


def test_track_keeps_id_when_object_moves_slightly():
    tracker = IoUTracker(iou_threshold=0.3)
    # Frame 1: one object.
    t1 = tracker.update([_det("car", (0, 0, 10, 10))])
    assert len(t1) == 1
    first_id = t1[0].track_id

    # Frame 2: same object shifted a little (still high overlap).
    t2 = tracker.update([_det("car", (1, 0, 11, 10))])
    assert len(t2) == 1
    assert t2[0].track_id == first_id  # SAME id -> tracked correctly
    assert t2[0].hits == 2


def test_new_nonoverlapping_object_gets_new_id():
    tracker = IoUTracker(iou_threshold=0.3)
    a = tracker.update([_det("car", (0, 0, 10, 10))])[0]
    # A detection far away cannot match the existing track.
    tracks = tracker.update([_det("car", (100, 100, 110, 110))])
    ids = {t.track_id for t in tracks}
    assert a.track_id not in ids  # the old track wasn't seen this frame
    assert len(ids) == 1 and a.track_id + 1 in ids  # got a fresh id
