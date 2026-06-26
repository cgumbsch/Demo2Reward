import h5py
import numpy as np
from PIL import Image

from .dataset import VLMCritic_Dataset, EpisodeData


def get_task_instruction(task):
    task_register = {
        'assembly': 'pick up a nut and place it onto a peg',
        'boxclose': 'grasp the cover and close the box with it',
        'coffeepush': 'push a mug under a coffee machine',
        'stickpull': 'grasp a stick and pull a box with the stick',
    }
    return task_register[task]


_MW_NAMES = {
    'coffeepush': 'CoffeePush', 'stickpull': 'StickPull',
    'assembly': 'Assembly', 'boxclose': 'BoxClose',
}


class MetaworldDataset(VLMCritic_Dataset):

    def __init__(self, data_root, task, ref_i=0, num_demos=3, num_test=5,
                 past_len=5, **kwargs):
        self._data_root = data_root
        self._task = task
        self._total = ref_i + num_demos + num_test
        super().__init__(ref_i=ref_i, num_demos=num_demos, num_test=num_test,
                         past_len=past_len, **kwargs)

    def _load_episodes(self):
        mw_name = _MW_NAMES[self._task]
        path = f'{self._data_root}/metaworld/{mw_name}/dataset.hdf5'

        images, successes = [], []
        with h5py.File(path, "r") as f:
            for i in range(self._total):
                frames = f[f'data/demo_{i}/obs/corner2_image'][:].transpose(0, 2, 3, 1)
                images.append([Image.fromarray(fr.astype(np.uint8)) for fr in frames])
                successes.append(list(f[f'data/demo_{i}/rewards'][:]))

        return EpisodeData(images=images, successes=successes)
