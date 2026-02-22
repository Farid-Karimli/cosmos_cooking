import math
from pathlib import Path

import numpy as np

from video_utils import trim_video


def get_step_annotations(recording_id: str, step_annotations: list[dict]) -> list[dict] | None:
    for obj in step_annotations:
        if obj["recording_id"] == recording_id:
            return obj["steps"]
    return None


def get_error_descriptions_for_step(
    recording_id: str,
    step_id: int,
    error_annotations: list[dict],
) -> str:
    """
    Get error tag and description strings for a step from error_annotations (e.g. error_annotations.json).
    Returns a single string with all errors for that step (e.g. "Missing Step: Skipped this step").
    """
    for rec in error_annotations:
        if rec.get("recording_id") != recording_id:
            continue
        for step in rec.get("step_annotations", []):
            if step.get("step_id") == step_id and "errors" in step:
                parts = []
                for err in step["errors"]:
                    tag = err.get("tag", "")
                    desc = err.get("description", "")
                    if tag and desc:
                        parts.append(f"{tag}: {desc}")
                    elif desc:
                        parts.append(desc)
                return "; ".join(parts) if parts else ""
    return ""

def prepare_data_for_task1(recording_id: str, video_path: str, step_annotations: list[dict], error_annotations: list[dict]) -> tuple[str, str, str]:
    """
    Prepare data for error recognition task, prompted with recipe instructions.

    Args:
        recording_id: Id of the recording.
        video_path: Path to the video file.
        step_annotations: Step annotations for the video (e.g. from downloaded_video_annotations.json).
        error_annotations: Error annotations with per-step error descriptions (e.g. from
            downloader/metadata/error_annotations.json). Used to include error tag and description.

    Returns:
        Tuple of (path to trimmed video, recipe text for the relevant steps, error description).
        error_description is non-empty only when the video has errors (first error step's
        tag and description from error_annotations); otherwise "".

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
        # Case 1: Has errors — trim from start to first error + 5 seconds; get error description
        recipe_steps = steps[: first_error_idx + 1]
        step_id = steps[first_error_idx].get("step_id")
        if step_id is not None and error_annotations:
            error_description = get_error_descriptions_for_step(recording_id, step_id, error_annotations)
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
    p = Path(video_path)
    output_path = str(p.parent / (p.stem + "_trimmed" + p.suffix))
    trimmed_video = trim_video(video_path, start_time, end_time, output_path)
    return trimmed_video, recipe, error_description


def prepare_data_for_task2(recording_id: str, video_path: str, step_annotations: list[dict], error_annotations: list[dict]) -> tuple[str, str]:
    """
    Same video trimming as prepare_data_for_task1_prompted, but without building the recipe.
    Returns only the path to the trimmed video and the error description (if any).

    Args:
        recording_id: Id of the recording.
        video_path: Path to the video file.
        step_annotations: Step annotations for the video (e.g. from downloaded_video_annotations.json).
        error_annotations: Error annotations (e.g. from downloader/metadata/error_annotations.json).

    Returns:
        Tuple of (path to trimmed video, error description). error_description is non-empty
        only when the video has errors; otherwise "".

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
        if step_id is not None and error_annotations:
            error_description = get_error_descriptions_for_step(recording_id, step_id, error_annotations)
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

    p = Path(video_path)
    output_path = str(p.parent / (p.stem + "_trimmed" + p.suffix))
    trimmed_video = trim_video(video_path, start_time, end_time, output_path)
    return trimmed_video, error_description


if __name__ == "__main__":
    import json

    recording_id = "29_22"
    base = Path(__file__).resolve().parent.parent
    video_path = str(base / "captain_cook_4d" / "gopro" / "resolution_360p" / f"{recording_id}_360p.mp4")
    step_annotations = json.loads((base / "captain_cook_4d" / "gopro" / "resolution_360p" / "downloaded_video_annotations.json").read_text())
    error_annotations = json.loads((base / "downloader" / "metadata" / "error_annotations.json").read_text())
    trimmed_video, recipe, error_description = prepare_data_for_task1_prompted(recording_id, video_path, step_annotations, error_annotations)
    print(trimmed_video)
    print(recipe)
    if error_description:
        print("Error:", error_description)