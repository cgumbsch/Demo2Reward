"""Lightweight logging for Demo2Reward.

Writes scalar metrics and instructions to TensorBoard, dumps the best
instructions to a text file, and optionally prints to stdout. Weights & Biases
logging is available as an opt-in extra (``use_wandb=True``); when enabled,
TensorBoard scalars are mirrored to W&B and the project / entity / run name are
supplied by the caller (e.g. via command-line arguments).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, Optional

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from .evaluate_prompts import compute_score


class Logger(ABC):
    """Abstract base class for logging."""

    @abstractmethod
    def write(self, step: int, prompt: str, stats: Dict[str, float],
              score: float, mode: str, verbose: bool) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class TensorBoardLogger(Logger):
    """Logs metrics/instructions to TensorBoard and a local text file.

    Optionally also logs to Weights & Biases when ``use_wandb=True``.
    """

    def __init__(self, target_path, name=None, args=None,
                 use_wandb: bool = False,
                 wandb_project: str = "demo2reward",
                 wandb_entity: Optional[str] = None):
        super().__init__()
        os.makedirs(target_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=target_path)
        self._prompt_file = os.path.join(target_path, "instructions.txt")

        self._use_wandb = use_wandb
        if use_wandb:
            import wandb
            self._wandb = wandb
            wb_config = {
                k: (str(v) if not isinstance(v, (bool, int, float, str, type(None))) else v)
                for k, v in vars(args).items()
            } if args is not None else {}
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=name,
                sync_tensorboard=True,  # mirror TensorBoard scalars to W&B
                dir=target_path,
                config=wb_config,
                settings={"init_timeout": 180},
            )

    def write(self, step, prompt, stats, score, mode='train', verbose=False,
              log_prompt=True, moving_score=None):
        self.writer.add_scalar(f"{mode}/TPR", stats['TPR'], step)
        self.writer.add_scalar(f"{mode}/TNR", stats['TNR'], step)
        self.writer.add_scalar(f"{mode}/FPR", stats['FPR'], step)
        self.writer.add_scalar(f"{mode}/FNR", stats['FNR'], step)
        self.writer.add_scalar(f"{mode}/score", score, step)
        if moving_score is not None:
            self.writer.add_scalar(f"{mode}/moving_score", moving_score, step)
        self.writer.add_scalar(f"{mode}/sum_rates", compute_score(stats, "sum_rates"), step)
        self.writer.add_text(f"{mode}/instructions", prompt, step)
        self.writer.flush()

        if log_prompt:
            with open(self._prompt_file, "a") as f:
                f.write(f"[step {step}] [{mode}] score={score:.3f}\n{prompt}\n\n")
            if self._use_wandb:
                self._wandb.log({f"{mode}/prompt": self._wandb.Html(prompt)}, step=step)

        if verbose:
            print(f"++++ {step} ++++")
            print(f"{mode} instruction:")
            print(prompt)
            print(f"{mode} score: {score}")
            print(f"True Positive Rate {stats['TPR']:.1f}%")
            print(f"True Negative Rate {stats['TNR']:.1f}%")
            print(f"False Positive Rate {stats['FPR']:.1f}%")
            print(f"False Negative Rate {stats['FNR']:.1f}%")

    def log_ab_plot(self, plotname, step, a, b, a_label='', b_label='',
                    y_label="Reward", x_label="t", mode='train',
                    ylim_min=-0.1, ylim_max=1.1):
        fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
        assert len(a) == len(b)
        ax.plot(a, label=a_label or "a")
        ax.plot(b, label=b_label or "b")
        ax.set_ylim(ylim_min, ylim_max)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.writer.add_figure(f"{mode}/{plotname}", fig, global_step=step)
        plt.close(fig)
        self.writer.flush()

    def close(self):
        self.writer.flush()
        self.writer.close()
        if self._use_wandb:
            self._wandb.finish()
