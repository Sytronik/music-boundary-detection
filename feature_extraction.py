"""
feature_extraction.py

A file related with extracting feature.
For the baseline code it loads audio files and extract mel-spectrogram using Librosa.
Then it stores in the './feature' folder.
"""
import os
import numpy as np
import librosa
from pathlib import Path

from hparams import hparams


def melspectrogram(y, sr):
    S = librosa.stft(y,
                     n_fft=hparams.fft_size,
                     hop_length=hparams.hop_size,
                     win_length=hparams.win_size)

    if not hasattr(melspectrogram, 'mel_basis'):
        melspectrogram.mel_basis = librosa.filters.mel(sr,
                                                       n_fft=hparams.fft_size,
                                                       n_mels=hparams.num_mels)
    mel_S = np.dot(melspectrogram.mel_basis, np.abs(S))
    logmel_S = np.log10(1 + 10 * mel_S)
    logmel_S = logmel_S.T

    return logmel_S


def main():
    # TODO: feature extraction
    pass


if __name__ == '__main__':
    main()
