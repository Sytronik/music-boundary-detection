"""
feature_extraction.py

A file related with extracting feature.
For the baseline code it loads audio files and extract mel-spectrogram using Librosa.
Then it stores in the './feature' folder.
"""
import csv
from argparse import ArgumentParser
from itertools import product

import librosa
import numpy as np
from numpy import ndarray

from hparams import hparams


def melspectrogram(y: ndarray, sr: int) -> ndarray:
    """

    :param y: (k, n) or (n,). k is no. of audio channels.
    :param sr:
    :return: (k, F, T) or (F, T).
    """
    if y.ndim == 1:
        y = y.unsqueeze(0)  # (1, n)

    S = []
    for item_y in y:
        S += librosa.stft(item_y,
                          n_fft=hparams.fft_size,
                          hop_length=hparams.hop_size,
                          win_length=hparams.win_size)  # (F[linear], T)

    S = np.stack(S, axis=0)  # (k, F[linear], T)

    if not hasattr(melspectrogram, 'mel_basis'):
        melspectrogram.mel_basis = librosa.filters.mel(sr,
                                                       n_fft=hparams.fft_size,
                                                       n_mels=hparams.num_mels)
    mel_S = np.einsum('mf,kft->kmt', melspectrogram.mel_basis, np.abs(S))  # k, F[mel], T
    logmel_S = np.log10(1 + 10 * mel_S)
    logmel_S = logmel_S.squeeze()  # (1, F, T) -> (F, T)

    return logmel_S


def main(kind_data: str):
    path_metadata = hparams.path_dataset[kind_data] / 'metadata/metadata.csv'
    path_audio_dir = hparams.path_dataset[kind_data] / 'audio'
    path_annot_dir = hparams.path_dataset[kind_data] / 'annotations'

    pitchstep = (0, *hparams.pitchstep)
    noise_db = (None, *hparams.noise_db)
    max_F_rm = (0, *hparams.max_F_rm)

    songs = []
    sr = 44100
    T = 1 / sr

    f_metadata = path_metadata.open('r', newline='')
    for idx, l_meta in enumerate(csv.reader(f_metadata)):
        if idx == 0:
            continue
        song_id = int(l_meta[0])

        path_audio = path_audio_dir / f'{song_id}.mp3'
        if path_audio.exists():
            songs += song_id
        y, _ = librosa.load(str(path_audio), sr=sr, mono=False)  # 2, n
        length = y.shape[1]

        if kind_data == 'train':
            ys_pitch = {0: y}
            for step, db, F in product(pitchstep, noise_db, max_F_rm):
                # pitch shift
                if step not in ys_pitch:
                    ys_pitch[step] = librosa.effects.pitch_shift(y, sr, n_steps=step)
                y = ys_pitch[step]

                # adding noise
                if db is not None:
                    y += librosa.db_to_amplitude(db) * np.random.randn(*y.shape)

                mel = melspectrogram(y, sr)  # 2, F, T

                # SpecAugmentation
                if F != 0:
                    height = np.random.randint(1, F + 1)
                    f0 = np.random.randint(0, hparams.num_mels - height)
                    mel[..., f0:f0 + height, :] = 0

                np.save(hparams.path_feature[kind_data] / f'{song_id}.npy', mel)
        else:
            mel = melspectrogram(y, sr)
            np.save(hparams.path_feature[kind_data] / f'{song_id}.npy', mel)

        n_frames = np.arange(0, length, hparams.hop_size)
        t_frames = n_frames * T

        t_boundaries = []
        sections = []
        f_annot = (
                path_annot_dir / f'{song_id}/parsed/textfile1_uppercase.txt'
        ).open('r', newline='')
        for l_annot in csv.reader(f_annot, delimiter='\t'):
            t_boundaries += l_annot[0]
            sections += l_annot[1]
        f_annot.close()

        # TODO: boundary label / segmentation map per song basis / segmap by section names
        # boundary_labels = []
        # raw_segmap = []
        # i_boundary = 0
        # for i_frame in range(len(t_frames)):
        #     t_frames[i_frames]

    f_metadata.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('kind_data')
    args = parser.parse_args()
    main(args.kind_data)
