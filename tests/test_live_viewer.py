"""Tests for Live Viewer in src/ll/live_viewer.py."""
import pathlib
import pytest
from ll.live_viewer import run_viewer

def test_live_viewer_headless_random_run(tmp_path):
    """Test running 1 episode of live viewer in headless mode with random policy."""
    video_out = str(tmp_path / "test_demo.mp4")
    run_viewer(
        config="random",
        seed=0,
        episodes=1,
        headless=True,
        record_path=video_out,
    )
    assert pathlib.Path(video_out).exists()
    assert pathlib.Path(video_out).stat().st_size > 1000

