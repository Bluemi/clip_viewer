#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm

from clip_viewer.clip_model import MobileModel
from clip_viewer.data import get_video_paths, VideoFrames
from clip_viewer.viewer import LineViewer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=Path)
    return parser.parse_args()


def main():
    args = get_args()

    model = MobileModel()

    video_paths = get_video_paths(args.input_file)
    for path in video_paths:
        analyze_video(model, path)


def analyze_video(model: MobileModel, path: Path):
    embeddings, frames = embed_video(model, path)
    embeddings_2d = create_2d_embeddings(embeddings)

    viewer = LineViewer(embeddings_2d, frames)
    viewer.run()


def embed_video(model: MobileModel, path: Path):
    frames = list(VideoFrames(path, verbose=False))
    embeddings = [model.encode_image(frame).numpy() for frame in tqdm(frames)]
    return np.array(embeddings), frames


def create_2d_embeddings(embeddings: np.ndarray) -> np.ndarray:
    return TSNE().fit_transform(embeddings)


if __name__ == '__main__':
    main()

