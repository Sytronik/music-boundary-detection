import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union
from dataclasses import dataclass, field, asdict


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

    kind_annotation: str = 'uppercase'
    len_gaussian_kernel: int = 31
    train_multi_annot: bool = True

    # augmentation
    pitchstep: Tuple[int] = (0, -1, 1)
    noise_db: Tuple[int] = (None, -24)
    max_F_rm: Tuple[int] = (0, 15)
    bans: Dict[str, List[int]] = field(init=False)

    # summary path
    logdir: str = './runs/best'

    # Model Parameters
    model: Dict[str, Any] = field(init=False)

    # Training Parameters
    scheduler: Dict[str, Any] = field(init=False)
    train_ratio = 0.85
    batch_size: int = 4
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2

    # Device-dependent Parameters
    # 'cpu', 'cuda:n', the cuda device no., or the tuple of the cuda device no.
    device: Union[int, str, Sequence[str], Sequence[int]] = (0, 1, 2, 3)
    out_device: Union[int, str] = 3
    num_workers: int = 4

    def __post_init__(self):
        self.path_dataset = dict(train=Path('/salami-data-public'),
                                 test=Path('/soundlab-salami-test'))
        self.path_feature = dict(train=Path('/salami-data-public/feature'),
                                 test=Path('/soundlab-salami-test/feature'))

        self.bans = dict(pitchstep=[],
                         noise_db=[],
                         max_F_rm=[])

        self.model = dict(ch_base=8,
                          depth=4,
                          kernel_size=(3, 11),
                          stride=(1, 1),
                          )
        self.scheduler = dict(restart_period=2,
                              t_mult=2,
                              eta_threshold=1.5,
                              )

    def is_banned(self, f: Path):
        aug_coeffs = f.stem.split('_')[1:]
        for coeff, bans in zip(aug_coeffs, self.bans.values()):
            if coeff in str(bans):
                return True

        return False

    # Function for parsing argument and set hyper parameters
    def parse_argument(self, parser=None, print_argument=True):
        if not parser:
            parser = argparse.ArgumentParser()
        dict_self = asdict(self)
        for k in dict_self:
            parser.add_argument(f'--{k}', default='')

        args = parser.parse_args()
        for k in dict_self:
            parsed = getattr(args, k)
            if parsed == '':
                continue
            if type(dict_self[k]) == str:
                setattr(self, k, parsed)
            else:
                v = eval(parsed)
                if isinstance(v, type(dict_self[k])):
                    setattr(self, k, eval(parsed))

        if print_argument:
            print('-------------------------')
            print('Hyper Parameter Settings')
            print('-------------------------')
            for k, v in asdict(self).items():
                print(f'{k}: {v}')
            print('-------------------------')

        return args


hparams = HParams()
