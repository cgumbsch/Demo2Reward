"""Base class for Demo2Reward datasets.

Subclasses only implement ``_load_episodes()`` which returns raw PIL images
and per-frame success labels. All splitting into validation / test / reference
sets, example-video composition, and the getter methods live here.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image

from utils.images import (
    past_frames, all_past_frames, past_frames_random,
    index_to_frame, past_frames_single_video, all_past_frames_single_video,
)


# ---------------------------------------------------------------------------
# Data returned by _load_episodes()
# ---------------------------------------------------------------------------

@dataclass
class EpisodeData:
    """Container for raw episode data returned by subclasses.

    ``images`` is a list of episodes, each a list of PIL Images.
    ``successes`` holds the matching per-frame binary success labels. The base
    class slices them into validation / test / reference splits using
    ``ref_i``, ``num_demos`` and ``num_test``.
    """
    images: List[List[Image.Image]]
    successes: List[list]


# ---------------------------------------------------------------------------
# Sequence splitting
# ---------------------------------------------------------------------------

def _divide_episodes(images, successes, past_len, video_mode,
                     sparse_sample=False, sparse_sample_until=-1,
                     sparse_sample_mod=1, full_past=False, enhance_factor=0):
    """Split episodes into positive/negative frame indices.

    ``images`` and ``successes`` are lists of episodes, each a list of
    PIL Images / booleans respectively.  Returns the tuple
    ``(videos, pos_ids, neg_ids, pos_past_ids, neg_past_ids,
      pos_percs, neg_percs)``.
    """
    sample = all_past_frames if full_past else past_frames
    random_enhancement = enhance_factor > 1
    if random_enhancement:
        np.random.seed(420)
        assert not full_past

    videos = []
    pos_ids, neg_ids = [], []
    pos_past, neg_past = [], []
    pos_percs, neg_percs = [], []

    for seq_i, (ep_imgs, ep_suc) in enumerate(zip(images, successes)):
        video = list(ep_imgs)       # already PIL
        videos.append(video)
        success_reached = False
        n_frames = len(video)

        for t, (frame, suc) in enumerate(zip(video, ep_suc)):
            if suc:
                success_reached = True
            suc_eff = success_reached if (success_reached and video_mode) else suc

            if sparse_sample and not suc_eff:
                if t < sparse_sample_until and t % sparse_sample_mod == 0:
                    continue

            pct = t / n_frames
            if suc_eff:
                pos_ids.append((seq_i, t))
                pos_past.append(sample(seq_i, t, past_len))
                pos_percs.append(pct)
                if random_enhancement:
                    for _ in range(enhance_factor - 1):
                        pos_ids.append((seq_i, t))
                        pos_past.append(past_frames_random(seq_i, t, past_len))
                        pos_percs.append(pct)
            else:
                neg_ids.append((seq_i, t))
                neg_past.append(sample(seq_i, t, past_len))
                neg_percs.append(pct)
                if random_enhancement:
                    for _ in range(enhance_factor - 1):
                        neg_ids.append((seq_i, t))
                        neg_past.append(past_frames_random(seq_i, t, past_len))
                        neg_percs.append(pct)

    return videos, pos_ids, neg_ids, pos_past, neg_past, pos_percs, neg_percs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_video_mode(raw_success, video_mode):
    """Return processed success list. If *video_mode* is True, once success
    is reached all subsequent frames are marked as successful."""
    result = []
    reached = False
    for s in raw_success:
        if s:
            reached = True
        result.append(reached if (reached and video_mode) else s)
    return result


def _make_percentage(n):
    """Return ``[0.0, ..., 1.0]`` of length *n*."""
    if n <= 1:
        return [0.0] * n
    return [i / (n - 1) for i in range(n)]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class VLMCritic_Dataset(ABC):
    """Abstract base for Demo2Reward evaluation datasets.

    Subclasses implement :meth:`_load_episodes` only.
    """

    def __init__(
        self,
        *,
        ref_i: int = 0,
        num_demos: int = 3,
        num_test: int = 5,
        video_mode: bool = False,
        past_len: int = 5,
        full_past: bool = False,
        enhance_data_factor: int = 0,
        sparse_videos_until: int = -1,
        sparse_sample_mod: int = 10,
    ):
        self._past_len = past_len
        self._full_past = full_past
        self._video_mode = video_mode
        self._sparse = sparse_videos_until > 0
        self._sparse_until = sparse_videos_until
        self._sparse_mod = sparse_sample_mod

        data = self._load_episodes()
        self._init_index_based(data, ref_i, num_demos, num_test,
                               past_len, video_mode, full_past,
                               enhance_data_factor)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_index_based(self, data, ref_i, num_demos, num_test,
                          past_len, video_mode, full_past, enhance_factor):
        images = data.images
        successes = data.successes

        # Reference demos
        self._reference_videos = []
        self._reference_successes = []
        for i in range(ref_i, ref_i + num_demos):
            self._reference_videos.append(list(images[i]))
            self._reference_successes.append(_apply_video_mode(successes[i], video_mode))

        # Val example: second demo
        self._val_ref_video = list(images[ref_i + 1])
        self._val_ref_success = _apply_video_mode(successes[ref_i + 1], video_mode)
        self._val_ref_pct = _make_percentage(len(self._val_ref_success))

        # Test example: first test demo (may be empty)
        if num_test > 0:
            self._test_ref_video = list(images[ref_i + num_demos])
            self._test_ref_success = _apply_video_mode(successes[ref_i + num_demos], video_mode)
            self._test_ref_pct = _make_percentage(len(self._test_ref_success))
        else:
            self._test_ref_video = []
            self._test_ref_success = []
            self._test_ref_pct = []

        # Negative examples are not available for index-based datasets
        self._val_neg_video = []
        self._val_neg_success = []
        self._val_neg_pct = []
        self._test_neg_video = []
        self._test_neg_success = []
        self._test_neg_pct = []

        sparse_kw = dict(sparse_sample=self._sparse,
                         sparse_sample_until=self._sparse_until,
                         sparse_sample_mod=self._sparse_mod)

        # Val split
        (self._val_videos, self._pos_ids, self._neg_ids,
         self._pos_past, self._neg_past,
         self._pos_pct, self._neg_pct) = _divide_episodes(
            images[ref_i:ref_i + num_demos],
            successes[ref_i:ref_i + num_demos],
            past_len, video_mode, full_past=full_past,
            enhance_factor=enhance_factor, **sparse_kw)

        # Test split (no enhancement)
        (self._test_videos, self._test_pos_ids, self._test_neg_ids,
         self._test_pos_past, self._test_neg_past,
         self._test_pos_pct, self._test_neg_pct) = _divide_episodes(
            images[ref_i + num_demos:ref_i + num_demos + num_test],
            successes[ref_i + num_demos:ref_i + num_demos + num_test],
            past_len, video_mode, full_past=full_past, **sparse_kw)

    # ------------------------------------------------------------------
    # Abstract: subclasses implement this only
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_episodes(self) -> EpisodeData:
        """Load raw episodes and return an :class:`EpisodeData`."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helpers for building example video clips
    # ------------------------------------------------------------------

    def _build_example_clips(self, ref_video, ref_success):
        """Build sub-videos (past frames + current) for each frame in a
        reference video.  Returns ``(clips, success_list)``."""
        if not ref_video:
            return [], []
        clips = []
        for idx, frame in enumerate(ref_video):
            if self._full_past:
                past_idx = all_past_frames_single_video(idx)
            else:
                past_idx = past_frames_single_video(idx, self._past_len)
            clip = [ref_video[p] for p in past_idx]
            clip.append(frame)
            clips.append(clip)
        return clips, ref_success

    # ------------------------------------------------------------------
    # Helpers for yielding frames / videos from index lists
    # ------------------------------------------------------------------

    def _yield_frames(self, videos, ids):
        for fid in ids:
            yield index_to_frame(videos, fid)

    def _yield_videos(self, videos, ids, past_ids):
        for fid, pids in zip(ids, past_ids):
            clip = [index_to_frame(videos, p) for p in pids]
            clip.append(index_to_frame(videos, fid))
            yield clip

    # ------------------------------------------------------------------
    # Public API — lengths
    # ------------------------------------------------------------------

    def len_val_positives(self) -> int:
        return len(self._pos_ids)

    def len_val_negatives(self) -> int:
        return len(self._neg_ids)

    def len_test_positives(self) -> int:
        return len(self._test_pos_ids)

    def len_test_negatives(self) -> int:
        return len(self._test_neg_ids)

    # ------------------------------------------------------------------
    # Public API — frame iterators
    # ------------------------------------------------------------------

    def get_val_positive_frames(self):
        return self._yield_frames(self._val_videos, self._pos_ids)

    def get_val_negative_frames(self):
        return self._yield_frames(self._val_videos, self._neg_ids)

    def get_test_positive_frames(self):
        return self._yield_frames(self._test_videos, self._test_pos_ids)

    def get_test_negative_frames(self):
        return self._yield_frames(self._test_videos, self._test_neg_ids)

    # ------------------------------------------------------------------
    # Public API — video iterators
    # ------------------------------------------------------------------

    def get_val_positive_videos(self):
        return self._yield_videos(self._val_videos, self._pos_ids, self._pos_past)

    def get_val_negative_videos(self):
        return self._yield_videos(self._val_videos, self._neg_ids, self._neg_past)

    def get_test_positive_videos(self):
        return self._yield_videos(self._test_videos, self._test_pos_ids, self._test_pos_past)

    def get_test_negative_videos(self):
        return self._yield_videos(self._test_videos, self._test_neg_ids, self._test_neg_past)

    # ------------------------------------------------------------------
    # Public API — reference
    # ------------------------------------------------------------------

    def get_reference(self, i=0):
        return self._reference_videos[i], self._reference_successes[i]

    # ------------------------------------------------------------------
    # Public API — percentages
    # ------------------------------------------------------------------

    def get_val_positive_percentages(self):
        yield from self._pos_pct

    def get_val_negative_percentages(self):
        yield from self._neg_pct

    def get_test_positive_percentages(self):
        yield from self._test_pos_pct

    def get_test_negative_percentages(self):
        yield from self._test_neg_pct

    # ------------------------------------------------------------------
    # Public API — example videos (for plotting)
    # ------------------------------------------------------------------

    def get_val_positive_example_video(self):
        return self._build_example_clips(self._val_ref_video, self._val_ref_success)

    def get_val_negative_example_video(self):
        return self._build_example_clips(self._val_neg_video, self._val_neg_success)

    def get_test_positive_example_video(self):
        return self._build_example_clips(self._test_ref_video, self._test_ref_success)

    def get_test_negative_example_video(self):
        return self._build_example_clips(self._test_neg_video, self._test_neg_success)

    # ------------------------------------------------------------------
    # Public API — example percentages
    # ------------------------------------------------------------------

    def get_val_positive_example_percentage(self):
        return self._val_ref_pct

    def get_val_negative_example_percentage(self):
        return self._val_neg_pct

    def get_test_positive_example_percentage(self):
        return self._test_ref_pct

    def get_test_negative_example_percentage(self):
        return self._test_neg_pct
