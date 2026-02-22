import math
from pathlib import Path

import cv2
import numpy as np

from video_utils import trim_video


def get_step_annotations(recording_id: str, step_annotations: list[dict]) -> list[dict] | None:
    for obj in step_annotations:
        if obj["recording_id"] == recording_id:
            return obj["steps"]
    return None


def get_video_fps(video_path: str) -> float | None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 0:
        return None
    return float(fps)


def get_error_descriptions_for_step(
    recording_id: str,
    step_id: int | None,
    error_annotations: list[dict] | None,
) -> str:
    if step_id is None or not error_annotations:
        return ""

    for rec in error_annotations:
        if rec.get("recording_id") != recording_id:
            continue
        for step in rec.get("step_annotations", []):
            if step.get("step_id") != step_id or "errors" not in step:
                continue
            parts = []
            for err in step.get("errors", []):
                tag = err.get("tag", "")
                desc = err.get("description", "")
                if tag and desc:
                    parts.append(f"{tag}: {desc}")
                elif desc:
                    parts.append(desc)
                elif tag:
                    parts.append(tag)
            return "; ".join(parts)
    return ""


def prepare_data_for_task1(
    recording_id: str,
    video_path: str,
    step_annotations: list[dict],
    error_annotations: list[dict] | None = None,
) -> tuple[list[np.ndarray], str, str, float | None]:
    """
    Prepare data for error recognition task, prompted with recipe instructions.

    Args:
        recording_id: Id of the recording.
        video_path: Path to the video file.
        step_annotations: Step annotations for the video (e.g. from downloaded_video_annotations.json).
        error_annotations: Error annotations used to retrieve error descriptions.

    Returns:
        Tuple of (trimmed video frames, recipe text, error_description, source_video_fps).

    Video trimming:
        1. If the video has errors: from start until 5 seconds after the first error.
        2. If no errors: segment containing the first 4 steps — starts 5 seconds before
           the first step and ends 5 seconds after the last of those steps (users often
           take up to ~2 minutes to start cooking).
    """
    steps = get_step_annotations(recording_id, step_annotations)
    if steps is None:
        raise ValueError(f"No step annotations found for recording {recording_id}")

    # Find first step index that has an error
    first_error_idx = None
    for i, step in enumerate(steps):
        if step["has_errors"]:
            first_error_idx = i
            break

    error_description = ""
    if first_error_idx is not None:
        # Case 1: Has errors — trim from start to first error + 5 seconds
        recipe_steps = steps[: first_error_idx + 1]
        step_id = steps[first_error_idx].get("step_id")
        error_description = get_error_descriptions_for_step(recording_id, step_id, error_annotations)
        if not error_description:
            error_description = "Error present but description unavailable."
        start_time = 0
        end_time = steps[first_error_idx]["end_time"] + 5
        # Guard against invalid timestamps (e.g. -1)
        if end_time < 0:
            end_time = 0
        end_time = math.floor(end_time)
    else:
        # Case 2: No errors — first 2 steps; start 5 sec before first step, end 5 sec after last of those steps
        recipe_steps = steps[:2]
        if not recipe_steps:
            raise ValueError(f"No steps for recording {recording_id}")
        first_start = recipe_steps[0]["start_time"]
        last_end = recipe_steps[-1]["end_time"]
        start_time = max(0, first_start - 5) if first_start >= 0 else 0
        end_time = (last_end + 5) if last_end >= 0 else start_time
        start_time = math.floor(start_time)
        end_time = math.floor(end_time)

    recipe = "\n".join(f"{i + 1}. {s['description']}" for i, s in enumerate(recipe_steps))

    trimmed_video = trim_video(video_path, start_time, end_time)
    video_fps = get_video_fps(video_path)
    return trimmed_video, recipe, error_description, video_fps


def prepare_data_for_task2(
    recording_id: str,
    video_path: str,
    step_annotations: list[dict],
    error_annotations: list[dict] | None = None,
) -> tuple[list[np.ndarray], str, float | None]:
    """
    Same video trimming as prepare_data_for_task1_prompted, but without building the recipe.
    Returns only the path to the trimmed video and the error description (if any).

    Args:
        recording_id: Id of the recording.
        video_path: Path to the video file.
        step_annotations: Step annotations for the video (e.g. from downloaded_video_annotations.json).
        error_annotations: Error annotations (e.g. from downloader/metadata/error_annotations.json).

    Returns:
        Tuple of (trimmed video frames, error_description, source_video_fps).

    Video trimming: same as prepare_data_for_task1_prompted (first error + 5 sec, or first N steps with 5 sec buffer).
    """
    steps = get_step_annotations(recording_id, step_annotations)
    if steps is None:
        raise ValueError(f"No step annotations found for recording {recording_id}")

    first_error_idx = None
    for i, step in enumerate(steps):
        if step["has_errors"]:
            first_error_idx = i
            break

    error_description = ""
    if first_error_idx is not None:
        step_id = steps[first_error_idx].get("step_id")
        error_description = get_error_descriptions_for_step(recording_id, step_id, error_annotations)
        if not error_description:
            error_description = "Error present but description unavailable."
        start_time = 0
        end_time = steps[first_error_idx]["end_time"] + 5
        if end_time < 0:
            end_time = 0
        end_time = math.floor(end_time)
    else:
        segment_steps = steps[:2]
        if not segment_steps:
            raise ValueError(f"No steps for recording {recording_id}")
        first_start = segment_steps[0]["start_time"]
        last_end = segment_steps[-1]["end_time"]
        start_time = max(0, first_start - 5) if first_start >= 0 else 0
        end_time = (last_end + 5) if last_end >= 0 else start_time
        start_time = math.floor(start_time)
        end_time = math.floor(end_time)

    trimmed_video = trim_video(video_path, start_time, end_time)
    video_fps = get_video_fps(video_path)
    return trimmed_video, error_description, video_fps


if __name__ == "__main__":
    import json

    recording_id = "29_22"
    base = Path(__file__).resolve().parent.parent
    video_path = str(base / "captain_cook_4d" / "gopro" / "resolution_360p" / f"{recording_id}_360p.mp4")
    step_annotations = json.loads((base / "captain_cook_4d" / "gopro" / "resolution_360p" / "downloaded_video_annotations.json").read_text())
    error_annotations = json.loads((base / "downloader" / "metadata" / "error_annotations.json").read_text())
    frames, recipe, error_description, video_fps = prepare_data_for_task1(recording_id, video_path, step_annotations, error_annotations)
    print(f"Frames: {len(frames)} | has_errors={bool(error_description)} | fps={video_fps}")
    print(recipe)