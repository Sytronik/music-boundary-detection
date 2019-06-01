"""
data_manager.py

A file that loads saved features and convert them into PyTorch DataLoader.
"""
from pathlib import Path
from typing import Sequence, Callable, Dict, Any, Tuple, List, Union
import multiprocessing as mp
from copy import copy

import numpy as np
from numpy import ndarray
import torch
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
        pool_calc = mp.Pool(min(mp.cpu_count()-2, 6))
        with mp.Manager() as manager:
            queue_data = manager.Queue()
            pool_loader.starmap_async(cls._load_data,
                                      [(f, queue_data) for f in all_files])
            result = [None] * len(all_files)
            for idx in tqdm(range(len(all_files)), desc='mean', dynamic_ncols=True):
                data = queue_data.get()
                result[idx] = pool_calc.apply_async(
                    cls._calc_per_data,
                    (data, list_fn)
                )

        result = [item.get() for item in result]
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
            result = [None] * len(all_files)
            for idx in tqdm(range(len(all_files)), desc='std', dynamic_ncols=True):
                data = queue_data.get()
                result[idx] = pool_calc.apply_async(
                    cls._calc_per_data,
                    (data, (cls._sq_dev,), (mean,))
                )

        pool_loader.close()
        pool_calc.close()
        result = [item.get() for item in result]
        print()

        sum_sq_dev = np.sum([item[cls._sq_dev] for item in result], axis=0)

        std = np.sqrt(sum_sq_dev / (sum_size // sum_sq_dev.size) + 1e-5)
        print('Calculated Std')

        return cls(mean, std)

    def save(self, fname: Path):
        np.savez(fname, mean=self.mean.data[ndarray], std=self.std.data[ndarray])

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
    def __init__(self, kind_data: str, hparams, normalization=None):
        self._PATH = hparams.path_feature[kind_data]
        self.all_files = [f for f in self._PATH.glob('*.npy')]
        self.all_files = sorted(self.all_files)
        # self.all_files = list(np.random.permutation(self.all_files).tolist())
        self.all_y = dict(**np.load(self._PATH / f'{hparams.segmap}_maps.npz'))

        if kind_data == 'train':
            f_normconst = self._PATH / 'normconst.npz'
            if f_normconst.exists() and not hparams.refresh_normconst:
                self.normalization = Normalization(**np.load(f_normconst))
            else:
                self.normalization = Normalization.calc_const(self.all_files)
                self.normalization.save(f_normconst)
        else:
            assert normalization
            self.normalization = normalization

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        """

        :param idx:
        :return: (x, y, len_x)
            x: tensor with size (F, T)
            y: tensor with size (T,)
            len_x: an integer T
        """

        # open feature file and return
        fname = self.all_files[idx]
        x = torch.tensor(np.load(fname), dtype=torch.float32)
        y = torch.tensor(self.all_y[fname.name.split('_')[0]], dtype=torch.int64)
        len_x = x.shape[2]
        y = y[:len_x]
        return x, y, len_x

    def __len__(self):
        return len(self.all_files)

    @staticmethod
    def pad_collate(batch):
        # stack samples with padding
        batch_x = [item[0].permute(2, 0, 1) for item in batch]
        batch_x = pad_sequence(batch_x, batch_first=True)  # B, T, C, F
        batch_x = batch_x.permute(0, 2, 3, 1)  # B, C, F, T

        batch_y = [item[1] for item in batch]
        batch_y = pad_sequence(batch_y, batch_first=True)  # B, T

        # len_x = torch.tensor([item[2] for item in batch], dtype=torch.float32)
        # len_x = len_x.unsqueeze(-1)  # B, 1
        len_x = [item[2] for item in batch]

        return batch_x, batch_y, len_x

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
    # test_set = SALAMIDataset('test', hparams, salami.normalization)

    train_loader = DataLoader(train_set,
                              batch_size=hparams.batch_size,
                              shuffle=True,
                              drop_last=False,
                              num_workers=hparams.num_workers,
                              pin_memory=True,
                              collate_fn=SALAMIDataset.pad_collate)
    valid_loader = DataLoader(valid_set,
                              batch_size=hparams.batch_size,
                              shuffle=False,
                              drop_last=False,
                              num_workers=hparams.num_workers,
                              pin_memory=True,
                              collate_fn=SALAMIDataset.pad_collate)
    # test_loader = DataLoader(test_set,
    #                          batch_size=hparams.batch_size,
    #                          shuffle=False,
    #                          drop_last=False,
    #                          num_workers=hparams.num_workers,
    #                          pin_memory=True,
    #                          collate_fn=SALAMIDataset.pad_collate)

    # return train_loader, valid_loader, test_loader
    return train_loader, valid_loader, None
