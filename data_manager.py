"""
data_manager.py

A file that loads saved features and convert them into PyTorch DataLoader.
"""
import multiprocessing as mp
from copy import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import DataPerDevice


class Normalization:
    @staticmethod
    def _sum(a: ndarray) -> ndarray:
        return a.sum(axis=-1, keepdims=True)

    @staticmethod
    def _sq_dev(a: ndarray, mean_a: ndarray) -> ndarray:
        return ((a - mean_a)**2).sum(axis=-1, keepdims=True)

    @staticmethod
    def _load_data(fname: Union[str, Path], queue: mp.Queue) -> None:
        x = np.load(fname)
        queue.put(x)

    @staticmethod
    def _calc_per_data(data,
                       list_func: Sequence[Callable],
                       args: Sequence = None,
                       ) -> Dict[Callable, Any]:
        """ gather return values of functions in `list_func`

        :param list_func:
        :param args:
        :return:
        """

        if args:
            result = {f: f(data, arg) for f, arg in zip(list_func, args)}
        else:
            result = {f: f(data) for f in list_func}
        return result

    def __init__(self, mean, std):
        self.mean = DataPerDevice(mean)
        self.std = DataPerDevice(std)

    @classmethod
    def calc_const(cls, all_files: List[Path]):

        # Calculate summation & size (parallel)
        list_fn = (np.size, cls._sum)
        pool_loader = mp.Pool(2)
        pool_calc = mp.Pool(min(mp.cpu_count() - 2, 6))
        with mp.Manager() as manager:
            queue_data = manager.Queue()
            pool_loader.starmap_async(cls._load_data,
                                      [(f, queue_data) for f in all_files])
            result: List[mp.pool.AsyncResult] = []
            for _ in tqdm(range(len(all_files)), desc='mean', dynamic_ncols=True):
                data = queue_data.get()
                result.append(pool_calc.apply_async(
                    cls._calc_per_data,
                    (data, list_fn)
                ))

        result: List[Dict] = [item.get() for item in result]
        print()

        sum_size = np.sum([item[np.size] for item in result])
        sum_ = np.sum([item[cls._sum] for item in result], axis=0)
        mean = sum_ / (sum_size // sum_.size)

        print('Calculated Size/Mean')

        # Calculate squared deviation (parallel)
        with mp.Manager() as manager:
            queue_data = manager.Queue()
            pool_loader.starmap_async(cls._load_data,
                                      [(f, queue_data) for f in all_files])
            result: List[mp.pool.AsyncResult] = []
            for idx in tqdm(range(len(all_files)), desc='std', dynamic_ncols=True):
                data = queue_data.get()
                result.append(pool_calc.apply_async(
                    cls._calc_per_data,
                    (data, (cls._sq_dev,), (mean,))
                ))

        pool_loader.close()
        pool_calc.close()
        result: List[Dict] = [item.get() for item in result]
        print()

        sum_sq_dev = np.sum([item[cls._sq_dev] for item in result], axis=0)

        std = np.sqrt(sum_sq_dev / (sum_size // sum_sq_dev.size) + 1e-5)
        print('Calculated Std')

        return cls(mean, std)

    def save(self, fname: Path):
        np.savez(fname, mean=self.mean.data[ndarray], std=self.std.data[ndarray])

    # normalize and denormalize functions can accept a ndarray or a tensor.
    def normalize(self, a):
        return (a - self.mean.get_like(a)) / self.std.get_like(a)

    def normalize_(self, a):
        a -= self.mean.get_like(a)
        a /= self.std.get_like(a)

        return a

    def denormalize(self, a):
        return a * self.std.get_like(a) + self.mean.get_like(a)

    def denormalize_(self, a):
        a *= self.std.get_like(a)
        a += self.mean.get_like(a)

        return a


class SALAMIDataset(Dataset):
    def __init__(self, kind_data: str, hparams, **kwargs):
        self._PATH: Path = hparams.path_feature[kind_data]

        self.all_files: List[Path] = [
            f for f in self._PATH.glob('*.npy') if not hparams.is_banned(f)
        ]
        self.all_files = sorted(self.all_files)
        # self.all_files = list(np.random.permutation(self.all_files).tolist())

        self.all_ys = dict(**np.load(self._PATH / f'{hparams.output_type}.npz'))
        self.all_ys = {k: torch.tensor(v, dtype=torch.float32) for k, v in self.all_ys.items()}

        self.all_boundaries = dict(**np.load(self._PATH / 'boundary_indexes.npz'))
        for s_song_id, boundary_idx in self.all_boundaries.items():
            length = len(boundary_idx)
            boundary_interval = np.zeros((length + 1, 2))
            boundary_interval[1:, 0] = boundary_idx
            boundary_interval[:-1, 1] = boundary_idx
            boundary_interval[-1, 1] = len(self.all_ys[s_song_id])
            boundary_interval *= hparams.hop_size / hparams.sample_rate
            self.all_boundaries[s_song_id] = boundary_interval

        if kind_data == 'train':
            f_normconst = self._PATH / 'normconst.npz'
            if f_normconst.exists() and not hparams.refresh_normconst:
                self.normalization = Normalization(**np.load(f_normconst))
            else:
                self.normalization = Normalization.calc_const(self.all_files)
                self.normalization.save(f_normconst)

            self.num_classes = 2

            # Calculate weight per class using no. of samples per class
            self.class_weight = np.zeros(self.num_classes, dtype=np.float32)
            for y in self.all_ys.values():
                for label in y:
                    self.class_weight[int(label > 0)] += 1
            self.class_weight = self.class_weight.max() / self.class_weight
            self.class_weight = torch.from_numpy(self.class_weight)
        else:
            try:
                self.normalization = kwargs['normalization']
                self.num_classes = kwargs['num_classes']
            except KeyError:
                raise Exception(kwargs)

    def __getitem__(self, idx: int) -> Tuple:
        """

        :param idx:
        :return: (x, y, boundary_idx, len_x, song_id)
            x: tensor with size (F, T)
            y: tensor with size (T,)
            boundaries: ndarray with size (num_boundary, 2)
            len_x: an integer T
            song_id:
        """

        # open feature file and return
        f = self.all_files[idx]
        s_song_id = f.name.split('_')[0]

        x = torch.tensor(np.load(f), dtype=torch.float32)
        y = self.all_ys[s_song_id]
        boundaries = self.all_boundaries[s_song_id]
        T = x.shape[2]
        song_id = int(s_song_id)

        return x, y, boundaries, T, song_id

    def __len__(self):
        return len(self.all_files)

    @staticmethod
    def pad_collate(batch: List[Tuple]) -> Tuple:
        """

        :param batch:
        :return:
        """
        # stack samples with padding
        batch_x = [item[0].permute(2, 0, 1) for item in batch]
        batch_x = pad_sequence(batch_x, batch_first=True)  # B, T, C, F
        batch_x = batch_x.permute(0, 2, 3, 1)  # B, C, F, T

        batch_y = [item[1] for item in batch]
        batch_y = pad_sequence(batch_y, batch_first=True)  # B, T

        batch_boundaries = [item[2] for item in batch]

        Ts = [item[3] for item in batch]
        batch_ids = [item[4] for item in batch]

        return batch_x, batch_y, batch_boundaries, Ts, batch_ids

    @classmethod
    def split(cls, dataset, ratio: Sequence[float]) -> Sequence:
        """ Split the dataset into `len(ratio)` datasets.

        The sum of elements of ratio must be 1,
        and only one element can have the value of -1 which means that
        it's automaticall set to the value so that the sum of the elements is 1

        :type dataset: SALAMIDataset
        :type ratio: Sequence[float]

        :rtype: Sequence[Dataset]
        """
        if type(dataset) != cls:
            raise TypeError
        n_split = len(ratio)
        ratio = np.array(ratio)
        mask = (ratio == -1)
        ratio[np.where(mask)] = 0

        assert (mask.sum() == 1 and ratio.sum() < 1
                or mask.sum() == 0 and ratio.sum() == 1)
        if mask.sum() == 1:
            ratio[np.where(mask)] = 1 - ratio.sum()

        idx_data = np.cumsum(np.insert(ratio, 0, 0) * len(dataset.all_files),
                             dtype=int)
        result = [copy(dataset) for _ in range(n_split)]
        all_f_per = np.random.permutation(dataset.all_files).tolist()

        for ii in range(n_split):
            result[ii].all_files = all_f_per[idx_data[ii]:idx_data[ii + 1]]

        return result


# Function to load numpy data and normalize, it returns dataloader for train, valid, test
def get_dataloader(hparams):
    salami = SALAMIDataset('train', hparams)
    train_set, valid_set = SALAMIDataset.split(salami, (hparams.train_ratio, -1))
    test_set = SALAMIDataset('test', hparams,
                             normalization=salami.normalization,
                             num_classes=salami.num_classes,
                             )

    common_kwargs = dict(batch_size=hparams.batch_size,
                         drop_last=False,
                         num_workers=hparams.num_workers,
                         pin_memory=True,
                         collate_fn=SALAMIDataset.pad_collate)
    train_loader = DataLoader(train_set, shuffle=True, **common_kwargs)
    valid_loader = DataLoader(valid_set, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **common_kwargs)

    return train_loader, valid_loader, test_loader
    # return train_loader, valid_loader, None
