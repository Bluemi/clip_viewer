from pathlib import Path
from typing import List, Iterator

import cv2
import numpy as np
from tqdm import trange

VIDEO_SUFFICES = ['.mp4', '.mkv']


def video_to_images(path: Path, step_size: int = 15) -> List[np.ndarray]:
    """
    Extract frames from video. Skipping frames to reduce size.

    :param path: Path to video.
    :return: List of images.
    """
    images = []
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file: {path}")

    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if counter == 0:
            images.append(frame)
        counter = (counter + 1) % step_size

    return images


def get_video_paths(path: Path) -> List[Path]:
    """
    Recursively get all video paths in a directory.

    :param path: The directory to search in.
    :return: List of video paths.
    """
    if path.is_file():
        if path.suffix not in VIDEO_SUFFICES:
            raise ValueError(f"Invalid file type: {path.suffix}")
        return [path]

    paths = []
    for p in path.iterdir():
        if p.is_dir():
            paths.extend(get_video_paths(p))
        elif p.suffix in VIDEO_SUFFICES:
            paths.append(p)
    return paths


class VideoFrames:
    def __init__(self, video_path: Path | str, verbose: bool = False, step_size: int = 1):
        self.video_path = video_path
        self.verbose = verbose
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.step_size = step_size

        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video file {video_path}")

    def __len__(self):
        return self.num_frames

    def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def __iter__(self) -> Iterator[np.ndarray]:
        self.reset()

        iter_func = trange if self.verbose else range
        for frame_index in iter_func(self.num_frames):
            success, frame = self.cap.read()
            if not success:
                break
            if frame_index % self.step_size == 0:
                yield frame
