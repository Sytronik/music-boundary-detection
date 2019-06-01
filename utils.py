import contextlib
import os
from pathlib import Path
from typing import Callable, Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
import matplotlib.pyplot as plt


def draw_segmap(song_id: int, truth: ndarray, pred: ndarray):
    """

    :param truth: (T,)
    :param pred: (T,)
    :return:
    """
    color_pred = plt.cm.Set3(pred)
    color_truth = plt.cm.Set3(truth)
    pos_pred = np.arange(len(pred))
    pos_truth = np.arange(len(truth))

    fig = plt.figure(figsize=(700, 400))
    plt.subplot(2, 1, 1, title=f'{song_id} prediction')
    plt.bar(pos_pred, height=1, color=color_pred)
    plt.subplot(2, 1, 2, title=f'{song_id} truth')
    plt.bar(pos_truth, height=1, color=color_truth)
    fig.tight_layout()
    return fig


def print_to_file(fname: Union[str, Path], fn: Callable, args=None, kwargs=None):
    """ All `print` function calls in `fn(*args, **kwargs)`
      uses a text file `fname`.

    :param fname:
    :param fn:
    :param args: args for fn
    :param kwargs: kwargs for fn
    :return:
    """
    if fname:
        fname = Path(fname).with_suffix('.txt')

    if args is None:
        args = tuple()
    if kwargs is None:
        kwargs = dict()

    with (fname.open('w') if fname else open(os.devnull, 'w')) as file:
        with contextlib.redirect_stdout(file):
            fn(*args, **kwargs)


def convert(a, astype: type, device: Union[int, torch.device] = None):
    if astype == Tensor:
        if type(a) == Tensor:
            return a.to(device)
        else:
            return torch.as_tensor(a, dtype=torch.float32, device=device)
    elif astype == ndarray:
        if type(a) == Tensor:
            return a.cpu().numpy()
        else:
            return a
    else:
        raise ValueError(astype)


class DataPerDevice:
    __slots__ = ('data',)

    def __init__(self, data_np: ndarray):
        self.data = {ndarray: data_np}

    def __getitem__(self, typeOrtup):
        if type(typeOrtup) == tuple:
            _type, device = typeOrtup
        elif typeOrtup == ndarray:
            _type = ndarray
            device = None
        else:
            raise IndexError

        if _type == ndarray:
            return self.data[ndarray]
        else:
            if typeOrtup not in self.data:
                self.data[typeOrtup] = convert(self.data[ndarray].astype(np.float32),
                                               Tensor,
                                               device=device)
            return self.data[typeOrtup]

    def get_like(self, other):
        if type(other) == Tensor:
            return self[Tensor, other.device]
        else:
            return self[ndarray]
