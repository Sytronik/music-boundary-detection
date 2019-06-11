# Music boundary detection using U-Net

This works for [SALAMI v2.0 dataset](https://github.com/DDMAL/salami-data-public).

Hyperparameters and configurations are in `hparams.py`

## Feature Extraction

```shell
python feature_extraction.py KIND_DATA [--mode=MODE] [--num-workers=N]
```

- `KIND_DATA` can be `'train'` or `'test'`.
- `MODE` can be `'io'`, `'in'`, or `'out'` (means what feature will be processed).
  Default is `'io'`.
- `N` can be an integer from `1` to `cpu_count()`.
  Default is `cpu_count() - 1`.

## DNN Training and Testing

```shell
python train_test.py [--test=EPOCH] [--(hyperparameter name)=(python script)]
```

## Test Result Analysis

```shell
python analyze_test.py EPOCH [--song=ID]
```

## Requirements

- python >= 3.7 (or 3.6 with dataclasses backport)
- numpy
- matplotlib
- PyTorch >= 1.0
- tensorboardX >= 1.7
- librosa
- tqdm
- [torchsummary](https://github.com/sksq96/pytorch-summary)
- [mir_eval](https://craffel.github.io/mir_eval/)
