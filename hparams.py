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
from pathlib import Path


@dataclass
class HParams(object):
    """
    If you don't understand 'field(init=False)' and __post_init__,
    read python 3.7 dataclass documentation
    """
    # Dataset Settings
    path_dataset: Dict[str, Path] = field(init=False)
    path_feature: Dict[str, Path] = field(init=False)

    # Feature Parameters
    sample_rate: int = 44100
    fft_size: int = 2048
    win_size: int = 2048
    hop_size: int = 1024
    num_mels: int = 128
    refresh_normconst: bool = False
    len_gaussian_kernel: int = 31  # 51~101

    # augmentation
    # pitchstep: Tuple[int] = (0,)
    pitchstep: Tuple[int] = (0, -1, 1)
    # noise_db: Tuple[int] = (None,)
    noise_db: Tuple[int] = (None, -24, -30, -36)
    # max_F_rm: Tuple[int] = (0,)
    max_F_rm: Tuple[int] = (0, 9, 15)
    bans: Dict[str, List[int]] = field(init=False)
    s_bans: Dict[str, List[str]] = field(init=False)

    # summary path
    logdir: str = './runs/score'

    # Model Parameters
    model: Dict[str, Any] = field(init=False)

    # Training Parameters
    scheduler: Dict[str, Any] = field(init=False)
    train_ratio = 0.7
    batch_size: int = 2
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3

    # Device-dependent Parameters
    # 'cpu', 'cuda:n', the cuda device no., or the tuple of the cuda device no.
    device: Union[int, str, Sequence[str], Sequence[int]] = (0, 1)
    out_device: Union[int, str] = 1
    num_workers: int = 2

    def __post_init__(self):
        # self.dataset_path = dict(train='./SALAMI',
        self.path_dataset = dict(train=Path('/salami-data-public'),
                                 test=Path('/soundlab-salami-test'))
        self.path_feature = dict(train=Path('/salami-data-public/feature'),
                                 test=Path('/soundlab-salami-test/feature'))

        # self.bans = dict(pitchstep=[-1, 1],
        #                  noise_db=[-24, -30, -36],
        #                  max_F_rm=[9, 15])
        self.bans = dict(pitchstep=[-1, 1],
                         noise_db=[-30, -36],
                         max_F_rm=[9])

        self.s_bans = {k: [str(item) for item in v] for k, v in self.bans.items()}

        self.model = dict(ch_base=8,
                          depth=4,
                          kernel_size=(3, 11),
                          stride=(1, 1),
                          )
        self.scheduler = dict(restart_period=4,
                              t_mult=2,
                              eta_threshold=1.5,
                              )

    def is_banned(self, f: Path):
        aug_coeffs = f.stem.split('_')[1:]
        for coeff, bans in zip(aug_coeffs, self.s_bans.values()):
            if coeff in bans:
                return True

        return False

    # Function for parsing argument and set hyper parameters
    def parse_argument(self, parser=None, print_argument=True):
        if not parser:
            parser = argparse.ArgumentParser()
        for var in vars(self):
            # value = getattr(hparams, var)
            argument = f'--{var}'
            parser.add_argument(argument, type=str, default='')

        args = parser.parse_args()
        for var in vars(self):
            parsed = getattr(args, var)
            if parsed != '':
                setattr(hparams, var, eval(parsed))

        if print_argument:
            print('-------------------------')
            print('Hyper Parameter Settings')
            print('-------------------------')
            for var in vars(self):
                value = getattr(hparams, var)
                print(f'{var}: {value}')
            print('-------------------------')

        return args


hparams = HParams()
# hparams.parse_argument()
