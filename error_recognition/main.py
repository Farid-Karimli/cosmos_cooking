"""
Running batch error recognition on all videos in the config.VIDEO_DIRECTORY.

Usage:
uv run python -m error_recognition.main
"""

from error_recognition.process import prepare_data_for_task1, prepare_data_for_task2
from error_recognition.prompts import TASK1_PROMPT, TASK1_PROMPT2, TASK2_PROMPT
import config

from transformers import BatchFeature, Qwen3VLForConditionalGeneration, Qwen3VLProcessor, BatchFeature

import torch
import numpy as np
from tqdm import tqdm
import os
import json

MODEL_NAME = "nvidia/Cosmos-Reason2-2B"
DEVICE = torch.device("cuda")

model, processor = None, None

def load_model_and_processor() -> tuple[Qwen3VLForConditionalGeneration, Qwen3VLProcessor]:
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
    processor = Qwen3VLProcessor.from_pretrained(MODEL_NAME)

    model.to(torch.device("cuda"))

    return model, processor

def process_inputs(video: str | list[np.ndarray], recipe_instructions: str) -> BatchFeature:
    # TODO: Implement the video preparation here
    pass


def truncate_at_first_answer(text: str, end_marker: str = "</answer>") -> str:
    """Keep only up to and including the first complete answer block to avoid repetition loops."""
    idx = text.find(end_marker)
    if idx != -1:
        return (text[: idx + len(end_marker)]).strip()
    return text.strip()


def _model_response_to_lines(response: list) -> list[str]:
    """Convert model_response (list of one string) to list of lines for readable JSON."""
    if not response or not isinstance(response[0], str):
        return response
    return response[0].split("\n")


def results_for_json(results: list[dict]) -> list[dict]:
    """Convert results so model_response is stored as lines for readable JSON (no long lines).
    When loading from JSON, get full text with: \"\\n\".join(entry[\"model_response\"])."""
    out = []
    for r in results:
        row = dict(r)
        if "model_response" in row:
            row["model_response"] = _model_response_to_lines(row["model_response"])
        out.append(row)
    return out


def error_recognition_with_path(video_path: str, recipe_instructions: str) -> str:
    video_messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    },
    {"role": "user", "content": [
            {
                "type": "video", 
                "video": video_path,
                "fps": 4,
            },
            {"type": "text", "text": (
                    TASK1_PROMPT.replace("[paste recipe steps]", recipe_instructions)
                )
            },
            ]
        },
    ]
    inputs = processor.apply_chat_template(
        video_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=4,
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        repetition_penalty=1.15,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [truncate_at_first_answer(t) for t in output_text]

def error_recognition_with_video(video: list[np.ndarray], recipe_instructions: str, video_fps: float | None = None) -> bool:
    """
    Perform inference on raw video frames instead of a video file path.

    Args:
        video: List of video frames as numpy arrays.
        recipe_instructions: Recipe instructions as a string.
        video_fps: FPS of the source video. If provided, passed as video_metadata to avoid sampling warnings.

    Returns:
        Output text from the model.
    """
    if recipe_instructions is not None:
        prompt = TASK1_PROMPT2.replace("[recipe steps]", recipe_instructions)
    else:
        prompt = TASK2_PROMPT

    video_metadata = None
    if video_fps is not None:
        video_metadata = [
            {
                "total_num_frames": len(video),
                "fps": video_fps,
                "frames_indices": list(range(len(video))),
            }
        ]

    inputs = processor(
        videos=[video],
        text=prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=4,
        video_metadata=video_metadata,
    )

    inputs = inputs.to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        repetition_penalty=1.15,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    output_text = [truncate_at_first_answer(t) for t in output_text]

    return output_text
    

def run_error_recognition_task1(n_videos: int | None = None) -> list[dict]:
    """
    Run error recognition on videos in the config.VIDEO_DIRECTORY.
    """
    with open(config.STEP_ANNOTATION_JSON) as f:
        step_annotations = json.load(f)

    with open(config.ERROR_ANNOTATION_JSON) as f:
        error_annotations = json.load(f)

    video_files = [f for f in os.listdir(config.VIDEO_DIRECTORY) if f.endswith(".mp4")]
    if n_videos is not None and n_videos < len(video_files):
        video_files = video_files[:n_videos]
    result_json = []
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(config.VIDEO_DIRECTORY, video_file)
        video_id = '_'.join(video_file.split("_")[:2])
            
        frames, recipe_instructions, error_description, video_fps = prepare_data_for_task1(
            recording_id=video_id,
            video_path=video_path,
            step_annotations=step_annotations,
            error_annotations=error_annotations,
        )
        has_errors = bool(error_description)

        response = error_recognition_with_video(frames, recipe_instructions=recipe_instructions, video_fps=video_fps)
        result_json.append({
            'video_id': video_id,
            'video_path': video_path,
            'model_response': response,
            'recipe': recipe_instructions,
            'has_errors': has_errors,
            'error_description': error_description,
        })

    return result_json

def run_error_recognition_task2(n_videos: int | None = None) -> list[dict]:
    """
    Run error recognition on videos in the config.VIDEO_DIRECTORY.
    """
    with open(config.STEP_ANNOTATION_JSON) as f:
        step_annotations = json.load(f)

    with open(config.ERROR_ANNOTATION_JSON) as f:
        error_annotations = json.load(f)

    video_files = [f for f in os.listdir(config.VIDEO_DIRECTORY) if f.endswith(".mp4")]
    if n_videos is not None and n_videos < len(video_files):
        video_files = video_files[:n_videos]
    result_json = []

    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(config.VIDEO_DIRECTORY, video_file)
        video_id = '_'.join(video_file.split("_")[:2])

        frames, error_description, video_fps = prepare_data_for_task2(
            recording_id=video_id,
            video_path=video_path,
            step_annotations=step_annotations,
            error_annotations=error_annotations,
        )
        has_errors = bool(error_description)

        response = error_recognition_with_video(frames, recipe_instructions=None, video_fps=video_fps)
        result_json.append({
            'video_id': video_id,
            'video_path': video_path,
            'model_response': response,
            'has_errors': has_errors,
            'error_description': error_description,
        })

    return result_json

if __name__ == "__main__":
    model, processor = load_model_and_processor()
    N_VIDEOS = 30
    try:
        print(f"*** Running error recognition task 1 ***")
        results = run_error_recognition_task1(n_videos=N_VIDEOS)

        print(f"*** Saving results to {config.ERROR_RECOGNITION_TASK1_RESULTS_JSON} ***")
        if not os.path.exists(os.path.dirname(config.ERROR_RECOGNITION_TASK1_RESULTS_JSON)):
            os.makedirs(os.path.dirname(config.ERROR_RECOGNITION_TASK1_RESULTS_JSON), exist_ok=True)
        with open(config.ERROR_RECOGNITION_TASK1_RESULTS_JSON, "w") as f:
            json.dump(results_for_json(results), f, indent=4)
        print(f"*** Results saved to {config.ERROR_RECOGNITION_TASK1_RESULTS_JSON} ***")

        print(f"*** Running error recognition task 2 ***")
        results = run_error_recognition_task2(n_videos=N_VIDEOS)

        print(f"*** Saving results to {config.ERROR_RECOGNITION_TASK2_RESULTS_JSON} ***")
        if not os.path.exists(os.path.dirname(config.ERROR_RECOGNITION_TASK2_RESULTS_JSON)):
            os.makedirs(os.path.dirname(config.ERROR_RECOGNITION_TASK2_RESULTS_JSON), exist_ok=True)
        with open(config.ERROR_RECOGNITION_TASK2_RESULTS_JSON, "w") as f:
            json.dump(results_for_json(results), f, indent=4)
        print(f"*** Results saved to {config.ERROR_RECOGNITION_TASK2_RESULTS_JSON} ***")
    finally:
        del model, processor
        torch.cuda.empty_cache()