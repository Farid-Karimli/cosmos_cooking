EPISODIC_MEMORY_PROMPT = """
Look at this video: <|video_pad|>

The person in the video is cooking. 

After the video ends, where is the {associated_object}?

Answer in the following format:
<answer>
The {associated_object} is located [location].
</answer>
"""

EPISODIC_MEMORY_PROMPT2 = """
Look at this video: <|video_pad|>

The person in the video is cooking.

After the video ends, what is the person doing with the {associated_object}?
Answer in the following format:
<think>
- What is the person's interaction with the {associated_object}?
- What does this suggest about the object's last known location?
</think>

<answer>
The {associated_object} is located [inferred location].
</answer>
"""

