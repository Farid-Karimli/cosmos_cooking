ERROR_RECOGNITION_PROMPT = """ Take a look at this video: <|video_pad|>
and the following recipe instructions:

[paste recipe steps]

Note that the video does not involve the user following the entire recipe, so you should evaluate the user's performance up until the point where the video ends.

Question: Watch this video and determine if the person followed the recipe instructions correctly up until the end of the video. If not, explain what they did wrong.

Answer in the following format:
<think>Your reasoning about what you observe in the video and how it compares to the instructions up until the end of the video.</think>

<answer>
Followed correctly up until the end of the video: [Yes/No]
Error description: [if No, describe what went wrong up until the end of the video]
</answer>
"""