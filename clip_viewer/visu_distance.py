#!/usr/bin/env python3
import argparse
from itertools import batched
from pathlib import Path
from typing import List

import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm

from clip_viewer.models import SiglipModel, BaseEmbeddingModel
from clip_viewer.data import get_video_paths, VideoFrames
from clip_viewer.distance_viewer import DistanceViewer

QUERIES = ['fish', 'a fish', 'fish fin', 'a bottle']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=Path)
    return parser.parse_args()


def main():
    args = get_args()

    # model = MobileModel(traced=False, load_mcip=False)
    model = SiglipModel.load(True)

    video_paths = get_video_paths(args.input_file)
    analyse_videos(model, video_paths)


def analyse_videos(model: BaseEmbeddingModel, paths: List[Path]):
    embeddings = []
    frames = []
    for path in paths:
        emb, frms = embed_video(model, path)
        embeddings.append(emb)
        frames.extend(frms)
    embeddings = np.concatenate(embeddings, axis=0)

    text_embedding = model.encode_text(QUERIES, normalize=True).cpu().numpy()

    distances = np.dot(embeddings, text_embedding.T)

    viewer = DistanceViewer(distances, frames, QUERIES)
    viewer.run()


def embed_video(model: BaseEmbeddingModel, path: Path):
    frames = list(VideoFrames(path, verbose=False))
    embeddings = []
    for batch in tqdm(batched(tqdm(frames), 16)):
        embs = model.encode_image(list(batch), normalize=True).cpu().numpy()
        embeddings.extend(embs)
    return embeddings, frames


def create_2d_embeddings(embeddings: np.ndarray) -> np.ndarray:
    return TSNE().fit_transform(embeddings)


if __name__ == '__main__':
    main()

