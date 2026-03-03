from transformers import BatchFeature, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
import torch
import numpy as np

from utils import truncate_at_first_answer


MODEL_NAME = "nvidia/Cosmos-Reason2-2B"

def load_model_and_processor() -> tuple[Qwen3VLForConditionalGeneration, Qwen3VLProcessor]:
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
    processor = Qwen3VLProcessor.from_pretrained(MODEL_NAME)

    model.to(torch.device("cuda"))

    return model, processor

def process_inputs(video: str | list[np.ndarray], recipe_instructions: str) -> BatchFeature:
    # TODO: Implement
    pass

def run_inference(video: str | list[np.ndarray], prompt: str, model, processor) -> str:
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