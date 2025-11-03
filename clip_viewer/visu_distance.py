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
from clip_viewer.distance_viewer import DistanceViewer

QUERY = 'fish'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=Path)
    return parser.parse_args()


def main():
    args = get_args()

    model = MobileModel()

    video_paths = get_video_paths(args.input_file)
    analyse_videos(model, video_paths)


def analyse_videos(model: MobileModel, paths: List[Path]):
    embeddings = []
    frames = []
    for path in paths:
        emb, frms = embed_video(model, path)
        embeddings.append(emb)
        frames.extend(frms)
    embeddings = np.concatenate(embeddings, axis=0)

    text_embedding = model.encode_text(QUERY, normalize=True).cpu().numpy()

    distances = np.dot(embeddings, text_embedding)
    # distances = np.linalg.norm(embeddings - text_embedding, axis=1)

    viewer = DistanceViewer(distances, frames)
    viewer.run()


def embed_video(model: MobileModel, path: Path):
    frames = list(VideoFrames(path, verbose=False))
    embeddings = []
    for batch in tqdm(batched(tqdm(frames), 16)):
        embs = model.encode_image(list(batch), normalize=True).cpu().numpy()
        embeddings.extend(embs)
    return np.array(embeddings), frames


def create_2d_embeddings(embeddings: np.ndarray) -> np.ndarray:
    return TSNE().fit_transform(embeddings)


if __name__ == '__main__':
    main()

