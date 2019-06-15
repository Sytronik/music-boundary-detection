import contextlib
import os
from pathlib import Path
from typing import Callable, List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


def draw_lineplot(t_axis: ndarray, score: ndarray, prediction: ndarray, truth: ndarray,
                  song_id: int):
    fig, ax = plt.subplots(figsize=(len(score) // 1000, 2))

    ax.plot(t_axis, score, zorder=1)

    ax.vlines(x=prediction, ymin=0.7, ymax=1, colors='r', label='prediction', zorder=2)
    ax.vlines(x=truth, ymin=0, ymax=0.3, colors='y', label='truth', zorder=2)

    ax.set_title(str(song_id))
    ax.legend(loc='upper right')
    ax.set_yticks([0, 0.5, 1])
    ax.grid(True, axis='y')
    ax.set_xlabel('time (sec)')
    ax.set_ylabel('boundary score')

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


def convert(a: Union[Tensor, ndarray], astype: type, device=None):
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
