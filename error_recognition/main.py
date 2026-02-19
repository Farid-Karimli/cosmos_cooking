"""
Running batch error recognition on all videos in the config.VIDEO_DIRECTORY.

Usage:
uv run python -m error_recognition.main
"""

from error_recognition.process import prepare_data_for_task1_prompted
from error_recognition.prompts import ERROR_RECOGNITION_PROMPT
import config

from transformers import BatchFeature, Qwen3VLForConditionalGeneration, Qwen3VLProcessor, BatchFeature

import torch
import numpy as np
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
                    ERROR_RECOGNITION_PROMPT.replace("[paste recipe steps]", recipe_instructions)
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
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text

def error_recognition_with_video(video: list[np.ndarray], recipe_instructions: str) -> bool:
    """
    Perform inference on raw video frames instead of a video file path.

    Args:
        video: List of video frames as numpy arrays.
        recipe_instructions: Recipe instructions as a string.

    Returns:
        Output text from the model.
    """
    prompt = ERROR_RECOGNITION_PROMPT.replace("[paste recipe steps]", recipe_instructions)
    
    inputs = processor(
        videos = [video], 
        text = prompt, tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=4,
    )

    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text
    

def run_error_recognition() -> list[dict]:
    """
    Run error recognition on videos in the config.VIDEO_DIRECTORY.
    """
    with open(config.STEP_ANNOTATION_JSON) as f:
        step_annotations = json.load(f)

    result_json = []
    for video_path in os.listdir(config.VIDEO_DIRECTORY):
        if video_path.endswith(".mp4"):
            video_id = '_'.join(video_path.split("_")[:2])
            print(f"Video path: {video_path}, id: {video_id}")
            
            frames, recipe_instructions = prepare_data_for_task1_prompted(
                recording_id=video_id, 
                video_path=video_path,
                step_annotations=step_annotations,
                error_annotations=None
            )

            response = error_recognition_with_video(frames, recipe_instructions)
            result_json.append({
                'video_id': video_id,
                'video_path': video_path,
                'model_response': response,
                'recipe': recipe_instructions
            })

    return result_json


if __name__ == "__main__":
    model, processor = load_model_and_processor()
    try:
        results = run_error_recognition()

        if not os.path.exists(config.ERROR_RECOGNITION_TASK1_RESULTS_JSON):
            os.makedirs(os.path.dirname(config.ERROR_RECOGNITION_TASK1_RESULTS_JSON), exist_ok=True)
        
        with open(config.ERROR_RECOGNITION_TASK1_RESULTS_JSON, "w") as f:
            json.dump(results, f, indent=4)
    finally:
        del model, processor
        torch.cuda.empty_cache()