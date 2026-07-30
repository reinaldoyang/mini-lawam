import tempfile
from pathlib import Path

from mini_lawam.rollout_ur7e import (
    format_rollout_stem,
    next_trace_trial_number,
)


def test_missing_and_empty_trace_dirs_start_at_one():
    with tempfile.TemporaryDirectory() as parent:
        missing = Path(parent) / "traces"
        assert next_trace_trial_number(missing) == 1
        missing.mkdir()
        assert next_trace_trial_number(missing) == 1


def test_trace_numbering_uses_max_unique_trial_not_file_count():
    with tempfile.TemporaryDirectory() as trace_dir:
        root = Path(trace_dir)
        # Requested format: two files belonging to one logical trace.
        (root / "trial_10_20260724T133010_summary.json").touch()
        (root / "trial_10_20260724T133010_table_cam_first.png").touch()
        # Legacy format is recognized too, so migration cannot overwrite it.
        (root / "trial_007_async_20260723T120000_summary.json").touch()
        # Frame directories and unrelated files do not count as trace IDs.
        (root / "frames_trial_99_20260724T133010").mkdir()
        (root / "notes.txt").touch()

        assert next_trace_trial_number(root) == 11


def test_no_trace_dir_keeps_in_session_zero_based_numbering():
    assert next_trace_trial_number(None) == 0


def test_rollout_stem_matches_requested_format():
    # UTC/local timezone does not matter here because 0 formats deterministically
    # within a process; validate the structural contract rather than wall-clock text.
    stem = format_rollout_stem(10, 0.0)
    assert stem.startswith("trial_10_")
    assert "_async_" not in stem
    assert len(stem.rsplit("_", 1)[1]) == 15  # YYYYMMDDTHHMMSS


if __name__ == "__main__":
    test_missing_and_empty_trace_dirs_start_at_one()
    test_trace_numbering_uses_max_unique_trial_not_file_count()
    test_no_trace_dir_keeps_in_session_zero_based_numbering()
    test_rollout_stem_matches_requested_format()
    print("4 focused trace-numbering tests passed")
