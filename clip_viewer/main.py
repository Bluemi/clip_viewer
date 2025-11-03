#!/usr/bin/env python3
import argparse
from pathlib import Path

from clip_viewer.clip_model import MobileModel
from clip_viewer.data import get_video_paths, VideoFrames


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=Path)
    return parser.parse_args()


def main():
    args = get_args()

    model = MobileModel()

    video_paths = get_video_paths(args.input_file)
    for path in video_paths:
        for frame in VideoFrames(path, verbose=True):
            embedding = model.encode_image(frame)
            print(embedding.shape)

    print(model)


if __name__ == '__main__':
    main()

