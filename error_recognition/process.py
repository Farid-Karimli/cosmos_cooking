import numpy as np
from video_utils import trim_video
import math


def get_step_annotations(recording_id: str, step_annotations: list[dict]) -> dict:
    for obj in step_annotations:
        if obj['recording_id'] == recording_id:
            return obj['steps']
    return None

def prepare_data_for_task1_prompted(recording_id: str, video_path: str, step_annotations: list[dict], error_annotations: list[dict]) -> str:
    """
    Prepare data for error recognition task, prompted with recipe instructions.

    Args:
        video_path: Path to the video file.
        step_annotations: Step annotations for the video.
        error_annotations: Error annotations for the video.

    Note:
        The video is trimmed until 5 seconds after the first error.
        The recipe instructions are the step annotations concatenated together up until the first error.
    """
    steps = get_step_annotations(recording_id, step_annotations)
    if steps is None:
        raise ValueError(f"No step annotations found for recording {recording_id}")

    recipe = ''
    end_time = None

    # Find the first step with an error and accumulate recipe steps
    for i, step in enumerate(steps):
        if step['has_errors']:
            end_time = step['end_time'] if end_time == None else None

        recipe += f'{i+1}. {step['description']}'
        if i < len(steps) - 1:
            recipe += '\n'

    # Trim the video until 5 seconds after the first error or first 200 seconds if no errors
    if end_time is None:
        end_time = math.floor(60*2) # 2 minutes
    else:
        end_time = math.floor(end_time + 5)
    start_time = 0
    trimmed_video = trim_video(video_path, start_time, end_time)
    return trimmed_video, recipe