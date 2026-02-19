import numpy as np
from video_utils import trim_video
import math


def get_step_annotations(recording_id: str, step_annotations: list[dict]) -> dict:
    for obj in step_annotations:
        if obj['recording_id'] == recording_id:
            return obj['steps']
    return None


def get_recipe_instructions(recording_id: str, step_annotations: list[dict], end_time: float | None = None) -> str:
    steps = get_step_annotations(recording_id, step_annotations)
    if steps is None:
        raise ValueError(f"No step annotations found for recording {recording_id}")
    if end_time is None:
        return '\n'.join([step['description'] for step in steps])
    else:   
        return '\n'.join([step['description'] for step in steps if step['start_time'] <= end_time])

def prepare_data_for_task1(recording_id: str, video_path: str, step_annotations: list[dict]) -> tuple:
    """
    Prepare data for error recognition task, prompted with recipe instructions.

    Args:
        video_path: Path to the video file.
        step_annotations: Step annotations for the video.

    Note:
        The video is trimmed until 5 seconds after the first error.
        The recipe instructions are the step annotations concatenated together up until the first error.
        The trimmed video is returned along with the FPS.
    """
    steps = get_step_annotations(recording_id, step_annotations)
    if steps is None:
        raise ValueError(f"No step annotations found for recording {recording_id}")

    # full_recipe = get_recipe_instructions(recording_id, step_annotations)
    end_time = None
    has_errors = False

    # Find the first step with an error
    for i, step in enumerate(steps):
        if steps[i]['has_errors']:
            end_time = steps[i]['end_time'] if end_time == None else None
            has_errors = True
            recipe = get_recipe_instructions(recording_id, step_annotations, end_time=end_time)
            break

    # Trim the video until 5 seconds after the first error or first 2 minute if no errors
    if end_time is None:
        end_time = math.floor(60 * 2) # 2 minutes
        recipe = get_recipe_instructions(recording_id, step_annotations, end_time=end_time)
    else:
        end_time = math.floor(end_time + 5)
    start_time = 0
    frames, fps = trim_video(video_path, start_time, end_time)
    return frames, recipe, has_errors, fps


def prepare_data_for_task2(recording_id: str, video_path: str, step_annotations: list[dict]) -> tuple:
    """
    Prepare data for error recognition task, without recipe instructions.
    """
    steps = get_step_annotations(recording_id, step_annotations)
    if steps is None:
        raise ValueError(f"No step annotations found for recording {recording_id}")

    # Find the first step with an error
    end_time = None
    has_errors = False
    for step in steps:
        if step['has_errors']:
            end_time = step['end_time'] if end_time == None else None
            has_errors = True
            break

    # Trim the video until 5 seconds after the first error or first 2 minutes if no errors
    if end_time is None:
        end_time = math.floor(60 * 2) # 2 minutes
    else:
        end_time = math.floor(end_time + 5)

    start_time = 0
    frames, fps = trim_video(video_path, start_time, end_time)
    return frames, has_errors, fps