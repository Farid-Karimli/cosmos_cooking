import subprocess
from pathlib import Path

import cv2


def trim_video(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str | None = None,
) -> str:
    """
    Trim a video by start and end time in seconds.

    Args:
        video_path: Path to the input video.
        start_time: Start time in seconds (e.g. 65.0 for 1:05, 90.5 for 1:30.5).
        end_time: End time in seconds.
        output_path: Where to save the trimmed video. If None, writes next to
            the input with "_trimmed" before the extension.

    Returns:
        Path to the written file.
    """
    start_time = float(start_time)
    end_time = float(end_time)
    if output_path is None:
        p = Path(video_path)
        output_path = str(p.parent / f"{p.stem}_trimmed{p.suffix}")

    # Use ffmpeg with libx264 to avoid OpenCV/FFmpeg picking unavailable
    # hardware encoders (e.g. h264_v4l2m2m) and failing.
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_time),
                "-i",
                video_path,
                "-to",
                str(end_time - start_time),
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        return output_path
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

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
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()
    return output_path


def time_to_seconds(minutes: int | float = 0, seconds: float = 0) -> float:
    """Convert minutes and seconds to a single float (e.g. for trim_video)."""
    return float(minutes) * 60 + float(seconds)
