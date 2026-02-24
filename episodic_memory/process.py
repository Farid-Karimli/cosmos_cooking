import json
import os
from pathlib import Path
import numpy as np
import config
from video_utils import trim_video, stream_video_chunks

from episodic_memory.main import run_inference

OBJECTS = ["knife", "bowl", "cutting board", "spatula", "spoon"]
FPS = 30

def get_step_annotations(recording_id: str, step_annotations: list[dict]) -> dict:
    for obj in step_annotations:
        if obj['recording_id'] == recording_id:
            return obj['steps']
    return None

def get_associated_object(recording_id: str, str, step_annotations: list[dict], error_annotations: list[dict]) -> str:
    steps = get_step_annotations(recording_id, step_annotations)
    associated_object = None
    for i, step in enumerate(steps):
        description = step['description']
        if any(obj in description for obj in OBJECTS):
            associated_object = np.random.choice([obj for obj in OBJECTS if obj in description])
            break
    if associated_object is None:   
        raise ValueError(f"No associated object found for recording {recording_id}")
    return associated_object

def localize_object_occurrence(recording_id: str, video_path: str, step_annotations: list[dict], error_annotations: list[dict]) -> tuple[list[np.ndarray], str]:
    """
    Localize the first occurrence of the associated object in the video.

    Returns the start and end time of the occurrence.
    If no occurrence is found, returns None.

    Start time = time of first occurrence 
    End time = start time + 30 seconds. 
    """
    associated_object = get_associated_object(recording_id, video_path, step_annotations, error_annotations)


    for i, chunk in enumerate(stream_video_chunks(video_path, chunk_size=300)): # 300 frames = 10 seconds
        output_text = run_inference(chunk, f"Is there a {associated_object} in the video? If yes, return the frame number of the occurrence.")
        if "yes" in output_text[0].lower():
            start_time = i * 300 / FPS
            end_time = start_time + 300 / FPS
            return start_time, end_time
    return None

def prepare_data_for_episodic_memory(recording_id: str, video_path: str, step_annotations: list[dict], error_annotations: list[dict]) -> tuple[list[np.ndarray], str]:
    pass

if __name__ == "__main__":
    recording_id = "29_22"
    base = Path(__file__).resolve().parent.parent
    video_path = str(base / "captain_cook_4d" / "gopro" / "resolution_360p" / f"{recording_id}_360p.mp4")
    step_annotations = json.loads((base / "captain_cook_4d" / "gopro" / "resolution_360p" / "downloaded_video_annotations.json").read_text())

    associated_object = get_associated_object(recording_id, video_path, step_annotations, error_annotations=None)
    print(f"Associated object: {associated_object}")

    start_time, end_time = localize_object_occurrence(recording_id, video_path, step_annotations, error_annotations=None)
    print(f"Start time: {start_time}, End time: {end_time}")