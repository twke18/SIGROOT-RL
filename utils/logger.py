"""TensorBoard logging and video saving utilities."""
import os
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Wraps TensorBoard SummaryWriter and handles local video saving.

    Run directories: runs/<ALGO>_<YYYYMMDD_HHMMSS>/
    Video paths:     videos/<ALGO>_epoch<N>.mp4
    """

    def __init__(self, algo: str, log_dir: str = "runs", video_dir: str = "videos"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{algo}_{timestamp}"
        run_path = os.path.join(log_dir, run_name)
        os.makedirs(run_path, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=run_path)
        self.video_dir = video_dir
        self.algo = algo

    # ------------------------------------------------------------------
    # Scalar logging
    # ------------------------------------------------------------------

    def log_train_rollout(self, avg_reward: float, epoch: int):
        self.writer.add_scalar("train/rollout_avg_reward", avg_reward, epoch)

    def log_train_losses(self, actor_loss: float, critic_loss: float, epoch: int):
        self.writer.add_scalar("train/actor_loss", actor_loss, epoch)
        self.writer.add_scalar("train/critic_loss", critic_loss, epoch)

    def log_test(self, avg_reward: float, success_rate: float, epoch: int):
        self.writer.add_scalar("test/accumulated_reward", avg_reward, epoch)
        self.writer.add_scalar("test/success_rate", success_rate, epoch)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    # ------------------------------------------------------------------
    # Video saving
    # ------------------------------------------------------------------

    def save_video(self, frames: list[np.ndarray], epoch: int, fps: int = 30):
        """Save a list of RGB frames as an mp4 video file."""
        if not frames:
            return
        path = os.path.join(self.video_dir, f"{self.algo}_epoch{epoch:04d}.mp4")
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  Video saved: {path}")

    def close(self):
        self.writer.close()
