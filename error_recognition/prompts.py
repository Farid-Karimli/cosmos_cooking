ERROR_RECOGNITION_PROMPT = """
Recipe instructions:
[paste recipe steps]

Question: Watch this video and determine if the person followed the recipe instructions correctly. If not, explain what they did wrong.

Answer in the following format:
<think>Your reasoning about what you observe in the video and how it compares to the instructions.</think>

<answer>
Followed correctly: [Yes/No]
Error description: [if No, describe what went wrong]
</answer>
"""