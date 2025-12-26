"""Tests for video frame sampling utilities."""

from pathlib import Path

import numpy as np
import pytest

from src.image_utils import VideoProcessor


def _write_test_video(frames: list[np.ndarray], path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (width, height)
    )
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def test_iterate_frames_from_file_respects_stride(tmp_path):
    cv2 = pytest.importorskip("cv2")
    frames = [
        np.full((32, 32, 3), 0, dtype=np.uint8),
        np.full((32, 32, 3), 50, dtype=np.uint8),
        np.full((32, 32, 3), 100, dtype=np.uint8),
        np.full((32, 32, 3), 150, dtype=np.uint8),
    ]
    video_path = tmp_path / "sample.mp4"
    _write_test_video(frames, video_path)

    sampled = list(
        VideoProcessor.iterate_frames_from_file(
            str(video_path), frame_stride=2, max_frames=2
        )
    )

    assert len(sampled) == 2
    assert sampled[0].shape == (32, 32, 3)
    # MP4 codecs are typically lossy; allow small drift across platforms.
    assert abs(int(sampled[0][0, 0, 0]) - 0) <= 8
    assert abs(int(sampled[1][0, 0, 0]) - 100) <= 8


def test_sample_frames_from_bytes_round_trip(tmp_path):
    cv2 = pytest.importorskip("cv2")
    frames = [
        np.full((24, 24, 3), 10, dtype=np.uint8),
        np.full((24, 24, 3), 20, dtype=np.uint8),
        np.full((24, 24, 3), 30, dtype=np.uint8),
    ]
    video_path = tmp_path / "bytes.mp4"
    _write_test_video(frames, video_path)

    data = Path(video_path).read_bytes()
    sampled = VideoProcessor.sample_frames_from_bytes(
        data, suffix=".mp4", frame_stride=1, max_frames=5
    )

    assert len(sampled) == 3
    assert abs(int(sampled[0][0, 0, 0]) - 10) <= 8
    assert abs(int(sampled[-1][0, 0, 0]) - 30) <= 8
