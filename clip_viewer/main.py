#!/usr/bin/env python3
import argparse
from itertools import batched
from pathlib import Path
from typing import List

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
    analyse_videos(model, video_paths)
    # for path in video_paths:
    #     analyze_video(model, path)


def analyse_videos(model: MobileModel, paths: List[Path]):
    embeddings = []
    frames = []
    for path in paths:
        emb, frms = embed_video(model, path)
        embeddings.append(emb)
        frames.extend(frms)
    embeddings = np.concatenate(embeddings, axis=0)
    embeddings_2d = create_2d_embeddings(embeddings)

    viewer = LineViewer(embeddings_2d, frames)
    viewer.run()


def analyze_video(model: MobileModel, path: Path):
    embeddings, frames = embed_video(model, path)
    embeddings_2d = create_2d_embeddings(embeddings)

    viewer = LineViewer(embeddings_2d, frames)
    viewer.run()


def embed_video(model: MobileModel, path: Path):
    frames = list(VideoFrames(path, verbose=False))
    embeddings = []
    for batch in tqdm(batched(tqdm(frames), 16)):
        embs = model.encode_image(list(batch)).cpu().numpy()
        embeddings.extend(embs)
    return np.array(embeddings), frames


def create_2d_embeddings(embeddings: np.ndarray) -> np.ndarray:
    return TSNE().fit_transform(embeddings)


if __name__ == '__main__':
    main()

