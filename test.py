import transformers
import torch

import config
from video_utils import trim_video

model_name = "nvidia/Cosmos-Reason2-2B"
model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
    model_name, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
)
processor: transformers.Qwen3VLProcessor = (
    transformers.AutoProcessor.from_pretrained(model_name)
)
 
video_id = '29_22'
video_path = f"{config.VIDEO_DIRECTORY}/{video_id}_360p.mp4"
start_time = 0
end_time = 10
output_path = f"{config.VIDEO_DIRECTORY}/{video_id}_360p_trimmed.mp4"
trim_video(video_path, start_time, end_time, output_path)
print(f"Trimmed video saved to {output_path}")

video_messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    },
    {"role": "user", "content": [
            {
                "type": "video", 
                "video": output_path,
                "fps": 4,
            },
            {"type": "text", "text": (
                    "What is the activity in the video? Answer the question using the following format:\n\n<think>\nYour reasoning.\n</think>\n\nWrite your final answer immediately after the </think> tag."
                )
            },
        ]
    },
]

# Process inputs
inputs = processor.apply_chat_template(
    video_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    fps=4,
)
inputs = inputs.to(model.device)

# Run inference
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

print(output_text)