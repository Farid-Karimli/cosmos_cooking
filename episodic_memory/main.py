"""
Running batch episodic memory on all videos in the config.VIDEO_DIRECTORY.

Usage:
uv run python -m episodic_memory.main
"""

from error_recognition.main import truncate_at_first_answer
from error_recognition.process import prepare_data_for_task1_prompted
from error_recognition.prompts import ERROR_RECOGNITION_PROMPT
import config

from transformers import BatchFeature, Qwen3VLForConditionalGeneration, Qwen3VLProcessor, BatchFeature

import torch
import numpy as np
import os
import json

model = Qwen3VLForConditionalGeneration.from_pretrained("nvidia/Cosmos-Reason2-2B", torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
processor = Qwen3VLProcessor.from_pretrained("nvidia/Cosmos-Reason2-2B")

def process_inputs(video: str | list[np.ndarray], recipe_instructions: str) -> BatchFeature:
    # TODO: Implement
    pass

def run_inference(video: str | list[np.ndarray], prompt: str) -> str:
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
    output_text = [truncate_at_first_answer(t) for t in output_text]
    return output_text

def episodic_memory_with_video(video: list[np.ndarray], recipe_instructions: str) -> str:
    prompt = ERROR_RECOGNITION_PROMPT.replace("[paste recipe steps]", recipe_instructions)
    output_text = run_inference(video, prompt)
    return output_text
    

def run_episodic_memory() -> list[dict]:
    """
    Run episodic memory on videos in the config.VIDEO_DIRECTORY.
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

            response = episodic_memory_with_video(frames, recipe_instructions)
            result_json.append({
                'video_id': video_id,
                'video_path': video_path,
                'model_response': response,
                'recipe': recipe_instructions
            })

    return result_json


if __name__ == "__main__":
    results = run_episodic_memory()
    with open(config.EPISODIC_MEMORY_RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=4)