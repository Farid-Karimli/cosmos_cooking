import json
from pathlib import Path
import re
import numpy as np
import cv2
from video_utils import trim_video
from tqdm import tqdm

from .inference import run_inference

OBJECTS = ["knife", "bowl", "cutting board", "spatula", "spoon"]
FPS = 30
CHUNK_SECONDS = 30
POST_DISAPPEAR_SECONDS = 30
ABSENCE_CONFIRM_CHUNKS = 2

def get_step_annotations(recording_id: str, step_annotations: list[dict]) -> dict:
    for obj in step_annotations:
        if obj['recording_id'] == recording_id:
            return obj['steps']
    return None

def _extract_candidate_objects(recording_id: str, step_annotations: list[dict]) -> list[str]:
    steps = get_step_annotations(recording_id, step_annotations)
    if not steps:
        return OBJECTS.copy()

    scored_objects: list[tuple[int, str]] = []
    for obj in OBJECTS:
        earliest_idx = None
        for i, step in enumerate(steps):
            description = str(step.get("description", "")).lower()
            if obj in description:
                earliest_idx = i
                break
        if earliest_idx is not None:
            scored_objects.append((earliest_idx, obj))

    scored_objects.sort(key=lambda x: x[0])
    prioritized = [obj for _, obj in scored_objects]
    fallback = [obj for obj in OBJECTS if obj not in prioritized]
    return prioritized + fallback

def get_associated_object(recording_id: str, video_path: str, step_annotations: list[dict], error_annotations: list[dict]) -> str:
    del video_path, error_annotations
    candidates = _extract_candidate_objects(recording_id, step_annotations)
    if not candidates:
        raise ValueError(f"No associated object found for recording {recording_id}")
    return candidates[0]

def _get_video_metadata(video_path: str) -> tuple[float, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = total_frames / fps if fps > 0 else 0.0
    return float(fps), total_frames, duration

def _iter_video_chunks(video_path: str, chunk_size: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")

    start_frame = 0
    while True:
        frames = []
        for _ in range(chunk_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        if not frames:
            break
        yield start_frame, frames
        start_frame += len(frames)
    cap.release()

def _chunk_contains_object(frames: list[np.ndarray], obj_name: str, model, processor) -> bool:
    prompt = (
        'Look at this video: <|video_pad|>'
        f'Answer with ONLY "yes" or "no". '
        f'Is there a visible {obj_name} at any point in this clip?'
    )
    output = run_inference(frames, prompt, model, processor)
    answer = output[0].strip().lower() if output else ""
    return "yes" in answer

def _describe_last_known_location(frames: list[np.ndarray], obj_name: str) -> str | None:
    if not frames:
        return None
    prompt = (
        f'Answer with ONLY a short location phrase (max 8 words). '
        f'Where is the {obj_name} last seen in this clip? '
        f'Examples: "on the cutting board", "inside the bowl", "next to the stove". '
        f'If unclear, answer "unknown".'
    )
    output = run_inference(frames, prompt)
    if not output:
        return None
    raw = output[0].strip()
    cleaned = re.sub(r'["\'.]+', "", raw).strip().lower()
    if not cleaned or cleaned == "unknown":
        return None
    words = cleaned.split()
    return " ".join(words[:8])

def localize_object_occurrence(video_path: str, associated_object: str) -> tuple[float, float, float, float] | None:
    """
    Return (start_time, end_time, last_present_start_time, last_present_end_time)
    for the first present->absent occurrence.
    end_time is set to 30s after disappearance (capped by video end).
    """
    fps, _, duration = _get_video_metadata(video_path)
    chunk_size = max(1, int(round(CHUNK_SECONDS * fps)))
    n_chunks = duration / CHUNK_SECONDS


    chunk_starts: list[int] = []
    chunk_presence: list[bool] = []
    for start_frame, frames in _iter_video_chunks(video_path, chunk_size=chunk_size):
        is_present = _chunk_contains_object(frames, associated_object, model, processor)
        chunk_starts.append(start_frame)
        chunk_presence.append(is_present)

    if not chunk_presence:
        return None

    first_present_idx = next((i for i, present in enumerate(chunk_presence) if present), None)
    if first_present_idx is None:
        return None

    disappear_idx = None
    for i in range(first_present_idx + 1, len(chunk_presence)):
        if chunk_presence[i]:
            continue
        lookahead_end = min(len(chunk_presence), i + ABSENCE_CONFIRM_CHUNKS + 1)
        if all(not x for x in chunk_presence[i:lookahead_end]):
            disappear_idx = i
            break

    if disappear_idx is None:
        return None

    start_time = chunk_starts[first_present_idx] / fps
    disappearance_time = chunk_starts[disappear_idx] / fps
    end_time = min(disappearance_time + CHUNK_SECONDS + POST_DISAPPEAR_SECONDS, duration)
    if end_time <= start_time:
        return None
    last_present_idx = max(first_present_idx, disappear_idx - 1)
    last_present_start_time = chunk_starts[last_present_idx] / fps
    last_present_end_time = min(
        (chunk_starts[last_present_idx] + chunk_size) / fps,
        duration,
    )
    return start_time, end_time, last_present_start_time, last_present_end_time

def _trim_segment(video_path: str, start_time: float, end_time: float) -> list[np.ndarray]:
    frames = trim_video(video_path, start_time=start_time, end_time=end_time, output_path=None)
    if not isinstance(frames, list):
        raise ValueError("Expected trim_video to return frames when output_path is None.")
    return frames

def prepare_data_for_episodic_memory(
    recording_id: str,
    video_path: str,
    step_annotations: list[dict],
    error_annotations: list[dict], model, processor,
    include_last_known_location: bool = False,
) -> tuple[list[np.ndarray], str] | tuple[list[np.ndarray], str, str | None]:
    """
    Returns:
        frames: one segment where the chosen object appears then disappears, plus +30s.
        associated_object: the object to ask about at segment end.
    """
    del error_annotations
    candidates = _extract_candidate_objects(recording_id, step_annotations)
    for associated_object in candidates:
        localized = localize_object_occurrence(video_path=video_path, associated_object=associated_object, model=model, processor=processor)
        if localized is None:
            continue
        start_time, end_time, last_present_start_time, last_present_end_time = localized
        frames = _trim_segment(video_path, start_time, end_time)
        if not include_last_known_location:
            return frames, associated_object, start_time, end_time

        last_present_frames = _trim_segment(
            video_path,
            start_time=last_present_start_time,
            end_time=last_present_end_time,
        )
        last_known_location = _describe_last_known_location(last_present_frames, associated_object)
        return frames, associated_object, last_known_location

    raise ValueError(
        f"No valid object disappearance segment found for recording {recording_id}. "
        "Tried objects: " + ", ".join(candidates)
    )

if __name__ == "__main__":
    recording_id = "29_22"
    base = Path(__file__).resolve().parent.parent
    video_path = str(base / "captain_cook_4d" / "gopro" / "resolution_360p" / f"{recording_id}_360p.mp4")
    step_annotations = json.loads((base / "captain_cook_4d" / "gopro" / "resolution_360p" / "downloaded_video_annotations.json").read_text())

    frames, associated_object, last_known_location, start_time, end_time = prepare_data_for_episodic_memory(
        recording_id=recording_id,
        video_path=video_path,
        step_annotations=step_annotations,
        error_annotations=None,
        include_last_known_location=True,
    )
    print(f"Associated object: {associated_object}")
    print(f"Last known location: {last_known_location}")
    print(f"Prepared frames: {len(frames)}")
    print(f"Start time: {start_time:.2f}s, End time: {end_time:.2f}s")