"""
Running batch episodic memory on all videos in the config.VIDEO_DIRECTORY.

Usage:
uv run python -m episodic_memory.main
"""

from .prompts import EPISODIC_MEMORY_PROMPT
from .process import prepare_data_for_episodic_memory
from .inference import run_inference, load_model_and_processor

import config


import numpy as np
import json
import os
from tqdm import tqdm

model, processor = load_model_and_processor()

def episodic_memory_with_video(video: list[np.ndarray], associated_object: str) -> str:
    prompt = EPISODIC_MEMORY_PROMPT.format(associated_object=associated_object)
    output = run_inference(video, prompt, model, processor)
    answer = output[0].strip().lower() if output else ""
    return answer
    

def run_episodic_memory() -> list[dict]:
    """
    Run episodic memory on videos in the config.VIDEO_DIRECTORY.
    """
    results = []
    step_annotations = json.load(open(config.STEP_ANNOTATION_JSON))

    video_directory = config.VIDEO_DIRECTORY
    video_files = [f for f in os.listdir(video_directory) if f.endswith((".mp4", ".avi", ".mov"))]

    for video_file in tqdm(video_files, desc="Running episodic memory on videos", unit="videos"):
        video_path = os.path.join(video_directory, video_file)
        recording_id = os.path.splitext(video_file)[0].replace("_360p", "")

        frames, associated_object, last_known_location, start_time, end_time = prepare_data_for_episodic_memory(
            recording_id=recording_id,
            video_path=video_path,
            step_annotations=step_annotations,
            error_annotations=None,
            model=model,
            processor=processor,
        )

        if not frames:
            print(f"No frames extracted for {video_file}, skipping.")
            continue

        answer = episodic_memory_with_video(frames, associated_object)
        results.append({
            "recording_id": recording_id,
            "associated_object": associated_object,
            "start_time": start_time,
            "end_time": end_time,
            "answer": answer,
            "last_known_location": last_known_location,
        })

    return results


if __name__ == "__main__":
    results = run_episodic_memory()
    if not os.path.exists(os.path.dirname(config.EPISODIC_MEMORY_RESULTS_JSON)):
        os.makedirs(os.path.dirname(config.EPISODIC_MEMORY_RESULTS_JSON))

    with open(config.EPISODIC_MEMORY_RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=4)