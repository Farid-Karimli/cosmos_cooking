import subprocess
from pathlib import Path

import cv2

import numpy as np

import config


def trim_video(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str | None = None,
) -> str | list[np.ndarray]:
    """
    Trim a video by start and end time in seconds.

    Args:
        video_path: Path to the input video.
        start_time: Start time in seconds (e.g. 65.0 for 1:05, 90.5 for 1:30.5).
        end_time: End time in seconds.
        output_path: Where to save the trimmed video. If None, writes next to
            the input with "_trimmed" before the extension.

    Returns:
        Path to the written file if output_path is provided, otherwise a list of frames.
    """
    start_time = float(start_time)
    end_time = float(end_time)

    # Use ffmpeg with libx264 to avoid OpenCV/FFmpeg picking unavailable
    # hardware encoders (e.g. h264_v4l2m2m) and failing.
    # try:
    #     subprocess.run(
    #         [
    #             "ffmpeg",
    #             "-y",
    #             "-ss",
    #             str(start_time),
    #             "-i",
    #             video_path,
    #             "-to",
    #             str(end_time - start_time),
    #             "-c:v",
    #             "libx264",
    #             "-c:a",
    #             "aac",
    #             "-movflags",
    #             "+faststart",
    #             output_path,
    #         ],
    #         check=True,
    #         capture_output=True,
    #     )
    #     return output_path
    # except (FileNotFoundError, subprocess.CalledProcessError):
    #     pass

    # Fallback: OpenCV (may fail on systems without libx264 / with only HW encoder)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError(f"Could not get FPS from {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(round(start_time * fps))
    end_frame = int(round(end_time * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Prefer avc1 (software encoder when available)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    if output_path is not None:
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    else:
        out = None
    frame_idx = start_frame

    frames = []
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if out is not None:
            out.write(frame)
        else:
            frames.append(frame)

        frame_idx += 1
    cap.release()
    if out is not None:
        out.release()
    if output_path is not None:
        return output_path
    else:
        return frames


def time_to_seconds(minutes: int | float = 0, seconds: float = 0) -> float:
    """Convert minutes and seconds to a single float (e.g. for trim_video)."""
    return float(minutes) * 60 + float(seconds)

if __name__ == "__main__":
    videopath = config.VIDEO_DIRECTORY + "/27_26_360p.mp4"
    video_id = "27_26"
    start_time = 0
    end_time = 10

    output_path = config.VIDEO_DIRECTORY + f"/{video_id}_360p_trimmed.mp4"

    frames = trim_video(videopath, start_time, end_time, output_path)
    print(f"Trimmed video saved to {output_path}")


