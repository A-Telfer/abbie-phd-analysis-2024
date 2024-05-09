"""Extract frames from a folder of videos and save to an output directory

Copyright (c) 2021, M
"""
import logging
from pathlib import Path

import click
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.command()
@click.argument("input_folder", type=click.Path(exists=True))
@click.argument("output_folder", type=click.Path())
@click.option(
    "--frames-per-video",
    type=int,
    default=10,
    help="Number of frames to extract per video",
)
@click.option("--video-ext", type=str, default="MP4", help="Video file extension")
@click.option("--seed", type=int, default=42, help="Random seed")
def main(input_folder, output_folder, frames_per_video, video_ext, seed):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    video_files = [f for f in sorted(input_folder.glob(f"*.{video_ext}")) if not f.stem.startswith(".")]

    logger.info(f"Found {len(video_files)} video files")
    
    logger.info(f"Sampling {frames_per_video} frames from each video")
    np.random.seed(seed)
    frame_shape = None 
    for video_file in tqdm(video_files):
        cap = cv2.VideoCapture(str(video_file))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampled_frames = 0
        while sampled_frames < frames_per_video:
            frame_id = np.random.randint(0, total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                continue
            
            if frame_shape is None:
                frame_shape = frame.shape

            assert frame.shape == frame_shape, f"Frame shape mismatch: {frame.shape} vs {frame_shape}"
            frame = np.flip(frame, axis=2)
            frame = Image.fromarray(frame)
            frame.save(output_folder / f"{video_file.stem}_{frame_id:05}.png")
            sampled_frames += 1

        cap.release()

    logger.info(f"Saved frames to {output_folder}")



if __name__ == "__main__":
    main()