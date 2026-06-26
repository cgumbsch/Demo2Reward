import h5py
import numpy as np
from PIL import Image

from .dataset import VLMCritic_Dataset, EpisodeData


def get_task_instruction(task):
    rm_task_description = {
        'square': ('Two colored pegs (one square and one round) are mounted on the '
                   'tabletop, and a suare nut is placed on the table in front of a '
                   'single robot arm. The robot must fit the square nut onto the '
                   'square peg'),
        'can': ('A can is placed in a bin in front of a single robot arm. There are '
                'four containers next to the bin. The robot must place the can into '
                'its corresponding container'),
    }
    return rm_task_description[task]


class RobomimicDataset(VLMCritic_Dataset):

    def __init__(self, data_root, task, ref_i=0, num_demos=10, num_test=0,
                 past_len=5, **kwargs):
        self._data_root = data_root
        self._task = task
        self._total = ref_i + num_demos + num_test
        super().__init__(ref_i=ref_i, num_demos=num_demos, num_test=num_test,
                         past_len=past_len, **kwargs)

    def _load_episodes(self):
        path = f'{self._data_root}/robomimic/{self._task}/dataset.hdf5'

        images, successes = [], []
        with h5py.File(path, "r") as f:
            for i in range(self._total):
                frames = f[f'data/demo_{i}/obs/agentview_image'][:].transpose(0, 2, 3, 1)
                images.append([Image.fromarray(fr.astype(np.uint8)) for fr in frames])
                successes.append(list(f[f'data/demo_{i}/rewards'][:]))

        return EpisodeData(images=images, successes=successes)
