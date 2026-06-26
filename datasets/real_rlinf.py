import pickle

import torch
from torchvision.transforms.functional import to_pil_image

from .dataset import VLMCritic_Dataset, EpisodeData


def get_task_instruction(task):
    task_description = {
        'lid': 'cover the pot with the lid',
    }
    return task_description[task]


def _tensor_to_pil(tensor):
    t = tensor.detach().cpu()
    if t.dim() == 4 and t.size(0) == 1:
        t = t.squeeze(0)
    if t.dim() == 3 and t.shape[-1] in [1, 3]:
        t = t.permute(2, 0, 1)
    if t.dtype.is_floating_point:
        t = t.clamp(0, 1)
    return to_pil_image(t)


def _split_into_episodes(data):
    ep_imgs, ep_rews = [], []
    cur_imgs, cur_rews = [], []
    for d in data:
        cur_imgs.append(_tensor_to_pil(d['transitions']['obs']['main_images']))
        cur_rews.append(d['rewards'].detach().item())
        if d['dones'].detach().item():
            ep_imgs.append(cur_imgs)
            ep_rews.append(cur_rews)
            cur_imgs, cur_rews = [], []
    return ep_imgs, ep_rews


_PATHS = {
    'lid': '{root}/real/lid/dataset.pkl',
}


class RealRLInfDataset(VLMCritic_Dataset):

    def __init__(self, data_root, task, ref_i=0, num_demos=10, num_test=0,
                 past_len=2, **kwargs):
        self._data_root = data_root
        self._task = task
        self._total = ref_i + num_demos + num_test
        super().__init__(ref_i=ref_i, num_demos=num_demos, num_test=num_test,
                         past_len=past_len, **kwargs)

    def _load_episodes(self):
        path = _PATHS[self._task].format(root=self._data_root)
        with open(path, 'rb') as f:
            full_data = pickle.load(f)

        all_imgs, all_rews = _split_into_episodes(full_data)
        print("Found", len(all_imgs), "episodes")

        images = [all_imgs[i] for i in range(self._total)]
        successes = [all_rews[i] for i in range(self._total)]
        return EpisodeData(images=images, successes=successes)
