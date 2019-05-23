"""
hparams.py

A file sets hyper parameters for feature extraction and training.
You can change parameters using argument.
For example:
 $ python train_test.py --device=1 --batch_size=32.
"""
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Union, Sequence, Any


@dataclass
class HParams(object):
    # Dataset Settings
    dataset_path: Dict[str, str] = field(init=False)
    feature_path: Dict[str, str] = field(init=False)

    # Feature Parameters
    sample_rate: int = 22050
    fft_size: int = 1024
    win_size: int = 1024
    hop_size: int = 256
    num_mels: int = 128
    refresh_normconst: bool = False

    # summary path
    log_dir = './runs/main'

    # Model Parameters
    model: Dict[str, Any] = field(init=False)

    # Training Parameters
    scheduler: Dict[str, Any] = field(init=False)
    out_device: Union[int, str] = 3
    train_ratio = 0.7
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2

    # Device-dependent Parameters
    # 'cpu', 'cuda:n', the cuda device no., or the tuple of the cuda device no.
    device: Union[int, str, Sequence[str], Sequence[int]] = (0, 1, 2, 3)
    num_workers: int = 2

    def __post_init__(self):
        # self.dataset_path = dict(train='./SALAMI',
        self.dataset_path = dict(train='/SALAMI',
                                 test='/SOUNDLAB_MBD')
        self.feature_path = dict(train='./SALAMI_melspec',
                                 test='./SOUNDLAB_MBD_melspec')
        # TODO
        self.model = {}
        self.scheduler = {}

    # Function for parsing argument and set hyper parameters
    def parse_argument(self, print_argument=True):
        parser = argparse.ArgumentParser()
        for var in vars(self):
            value = getattr(hparams, var)
            argument = f'--{var}'
            parser.add_argument(argument, type=type(value), default=value)

        args = parser.parse_args()
        for var in vars(self):
            setattr(hparams, var, getattr(args, var))

        if print_argument:
            print('-------------------------')
            print('Hyper Parameter Settings')
            print('-------------------------')
            for var in vars(self):
                value = getattr(hparams, var)
                print(f'{var}: {value}')
            print('-------------------------')


hparams = HParams()
hparams.parse_argument()
