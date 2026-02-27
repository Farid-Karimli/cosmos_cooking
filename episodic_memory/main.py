"""
Running batch episodic memory on all videos in the config.VIDEO_DIRECTORY.

Usage:
uv run python -m episodic_memory.main
"""

from utils import truncate_at_first_answer
import config

from transformers import BatchFeature, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

import torch
import numpy as np
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
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
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
    pass
    

def run_episodic_memory() -> list[dict]:
    """
    Run episodic memory on videos in the config.VIDEO_DIRECTORY.
    """
    pass


if __name__ == "__main__":
    results = run_episodic_memory()
    with open(config.EPISODIC_MEMORY_RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=4)