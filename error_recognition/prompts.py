TASK1_PROMPT = """ Take a look at this video: <|video_pad|>

The person in the video is attempting to follow these recipe steps:
[paste recipe steps]

Question: Did the person execute these specific steps correctly in the video? 

Answer in the following format:
<think>
1. What steps from the recipe are being performed in the video?
2. How does the execution compare to the instructions?
3. Are there any mistakes or deviations?
</think>

<answer>
Steps executed correctly: [Yes/No]
Error description: [if No, describe what went wrong]
</answer>
"""

TASK2_PROMPT = """ Take a look at this video: <|video_pad|>

Question: Describe what the person is doing in this video. Pay attention to their technique and execution.

<think>
- What task is being performed?
- How is the person executing it?
- Does anything seem unusual or incorrect about their technique?
</think>

<answer>
[Describe what you observe, including any unusual or potentially incorrect actions]
</answer>
"""
